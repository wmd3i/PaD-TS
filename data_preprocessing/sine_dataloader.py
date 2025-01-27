"""
Modifed based on code from https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/Data_utils/sine_dataset.py
"""

import os
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


class SineDataset(Dataset):
    def __init__(
        self,
        window=128,
        num=30000,
        dim=12,
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
        super(SineDataset, self).__init__()
        assert period in ["train", "test"], "period must be train or test."
        if period == "train":
            assert ~(predict_length is not None or missing_ratio is not None), ""

        self.pred_len, self.missing_ratio = predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = (
            style,
            distribution,
            mean_mask_length,
        )

        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)

        self.rawdata = self.sine_data_generation(
            no=num,
            seq_len=window,
            dim=dim,
            save2npy=save2npy,
            seed=seed,
            dir=self.dir,
            period=period,
        )
        self.auto_norm = neg_one_to_one
        self.samples = self.normalize(self.rawdata)
        self.var_num = dim
        self.sample_num = self.samples.shape[0]
        self.window = window

        self.period, self.save2npy = period, save2npy
        if period == "test":
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

    def normalize(self, rawdata):
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(rawdata)
        return data

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return data

    @staticmethod
    def sine_data_generation(
        no, seq_len, dim, save2npy=True, seed=123, dir="./", period="train"
    ):
        """Sine data generation.

        Args:
           - no: the number of samples
           - seq_len: sequence length of the time-series
           - dim: feature dimensions

        Returns:
           - data: generated data
        """
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        # Initialize the output
        data = list()
        # Generate sine data
        for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
            # Initialize each time-series
            temp = list()
            # For each feature
            for k in range(dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)

            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1) * 0.5
            # Stack the generated data
            data.append(temp)

        # Restore RNG.
        np.random.set_state(st0)
        data = np.array(data)
        if save2npy:
            np.save(
                os.path.join(dir, f"sine_ground_truth_{seq_len}_{period}.npy"), data
            )

        return data

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
            np.save(os.path.join(self.dir, f"sine_masking_{self.window}.npy"), masks)

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
