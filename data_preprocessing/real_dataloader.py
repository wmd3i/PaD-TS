"""
Modifed based on code from: https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/Data_utils/real_datasets.py
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy import io


from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# normalization functions


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


class CustomDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        window=64,
        proportion=0.8,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        predict_length=None,
        missing_ratio=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
    ):
        super(CustomDataset, self).__init__()
        assert period in ["train", "test"], "period must be train or test."
        if period == "train":
            assert ~(predict_length is not None or missing_ratio is not None), ""
        self.name, self.pred_len, self.missing_ratio = (
            name,
            predict_length,
            missing_ratio,
        )
        self.style, self.distribution, self.mean_mask_length = (
            style,
            distribution,
            mean_mask_length,
        )
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == "train" else inference
        if period == "test":
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(
                    os.path.join(
                        self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"
                    ),
                    self.unnormalize(test_data),
                )
            np.save(
                os.path.join(
                    self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"
                ),
                self.unnormalize(train_data),
            )
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(
                        os.path.join(
                            self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"
                        ),
                        unnormalize_to_zero_to_one(test_data),
                    )
                np.save(
                    os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"
                    ),
                    unnormalize_to_zero_to_one(train_data),
                )
            else:
                if 1 - proportion > 0:
                    np.save(
                        os.path.join(
                            self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"
                        ),
                        test_data,
                    )
                np.save(
                    os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"
                    ),
                    train_data,
                )

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=""):
        """Reads a single .csv"""
        df = pd.read_csv(filepath, header=0)
        if name == "etth":
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(
                os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks
            )

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == "test":
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class fMRIDataset(CustomDataset):
    def __init__(self, proportion=1.0, **kwargs):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=""):
        """Reads a single .csv"""
        data = io.loadmat(filepath + "/sim4.mat")["ts"]
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler


def noise_mask(
    X,
    masking_ratio,
    lm=3,
    mode="separate",
    distribution="geometric",
    exclude_feats=None,
):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == "geometric":  # stateful (Markov chain)
        if mode == "separate":  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(
                        X.shape[0], lm, masking_ratio
                    )  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(
                np.expand_dims(
                    geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1
                ),
                X.shape[1],
            )
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == "separate":
            mask = np.random.choice(
                np.array([True, False]),
                size=X.shape,
                replace=True,
                p=(1 - masking_ratio, masking_ratio),
            )
        else:
            mask = np.tile(
                np.random.choice(
                    np.array([True, False]),
                    size=(X.shape[0], 1),
                    replace=True,
                    p=(1 - masking_ratio, masking_ratio),
                ),
                X.shape[1],
            )

    return mask
