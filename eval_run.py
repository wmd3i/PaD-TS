import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from eval_utils.metric_utils import display_scores
from eval_utils.discriminative_metric import discriminative_score_metrics
from eval_utils.predictive_metric import predictive_score_metrics
import torch
from eval_utils.MMD import BMMD, cross_correlation_distribution, BMMD_Naive, VDS_Naive


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


def discriminative_score(dataname, iterations, fake_data, length=24):
    if dataname == "energy":
        ori_data = np.load(f"./OUTPUT/samples/energy_norm_truth_{length}_train.npy")
    elif dataname == "stock":
        ori_data = np.load(f"./OUTPUT/samples/stock_norm_truth_{length}_train.npy")
    elif dataname == "sine":
        ori_data = np.load(f"./OUTPUT/samples/sine_ground_truth_{length}_train.npy")
    elif dataname == "fmri":
        ori_data = np.load(f"./OUTPUT/samples/fmri_norm_truth_{length}_train.npy")
    elif dataname == "mujoco":
        ori_data = np.load(f"./OUTPUT/samples/mujoco_norm_truth_{length}_train.npy")
    else:
        raise NotImplementedError(f"Unkown dataname: {dataname}")
    fake_data = unnormalize_to_zero_to_one(fake_data)
    fake_data = fake_data[: ori_data.shape[0]]
    discriminative_score = []
    print(f"Fake data:", "min ", fake_data.min(), ", max ", fake_data.max())
    print(f"Real data:", "min ", ori_data.min(), ", max ", ori_data.max())
    for i in range(iterations):
        print(i)
        temp_disc, fake_acc, real_acc, values = discriminative_score_metrics(
            ori_data[:], fake_data[: ori_data.shape[0]]
        )
        discriminative_score.append(temp_disc)
        print(f"Iter {i}: ", temp_disc, ",", fake_acc, ",", real_acc, "\n")
    print(f"{dataname}:")
    display_scores(discriminative_score)

    return values


def predictive_score(dataname, iterations, fake_data, length=24):
    if dataname == "energy":
        ori_data = np.load(f"./OUTPUT/samples/energy_norm_truth_{length}_train.npy")
    elif dataname == "stock":
        ori_data = np.load(f"./OUTPUT/samples/stock_norm_truth_{length}_train.npy")
    elif dataname == "sine":
        ori_data = np.load(f"./OUTPUT/samples/sine_ground_truth_{length}_train.npy")
    elif dataname == "fmri":
        ori_data = np.load(f"./OUTPUT/samples/fmri_norm_truth_{length}_train.npy")
    elif dataname == "mujoco":
        ori_data = np.load(f"./OUTPUT/samples/mujoco_norm_truth_{length}_train.npy")
    else:
        raise NotImplementedError(f"Unkown dataname: {dataname}")
    fake_data = unnormalize_to_zero_to_one(fake_data)
    fake_data = fake_data[: ori_data.shape[0]]
    predictive_score = []
    print(f"Fake data:", "min ", fake_data.min(), ", max ", fake_data.max())
    print(f"Real data:", "min ", ori_data.min(), ", max ", ori_data.max())
    for i in range(iterations):
        temp_pred = predictive_score_metrics(ori_data, fake_data[: ori_data.shape[0]])
        predictive_score.append(temp_pred)
        print(i, " epoch: ", temp_pred, "\n")
    print(f"{dataname}:")
    display_scores(predictive_score)


def BMMD_score(dataname, fake_data, length=24):
    if dataname == "energy":
        ori_data = np.load(f"./OUTPUT/samples/energy_norm_truth_{length}_train.npy")
    elif dataname == "stock":
        ori_data = np.load(f"./OUTPUT/samples/stock_norm_truth_{length}_train.npy")
    elif dataname == "sine":
        ori_data = np.load(f"./OUTPUT/samples/sine_ground_truth_{length}_train.npy")
    elif dataname == "fmri":
        ori_data = np.load(f"./OUTPUT/samples/fmri_norm_truth_{length}_train.npy")
    elif dataname == "mujoco":
        ori_data = np.load(f"./OUTPUT/samples/mujoco_norm_truth_{length}_train.npy")
    else:
        raise NotImplementedError(f"Unkown dataname: {dataname}")
    fake_data = unnormalize_to_zero_to_one(fake_data)
    fake_data = fake_data[: ori_data.shape[0]]
    ori_data = torch.tensor(ori_data).float()
    fake_data = torch.tensor(fake_data).float()

    ori_data = cross_correlation_distribution(ori_data).unsqueeze(-1).permute(1, 0, 2)
    fake_data = cross_correlation_distribution(fake_data).unsqueeze(-1).permute(1, 0, 2)

    assert ori_data.shape == fake_data.shape

    mmd_loss = BMMD(ori_data, fake_data, "rbf").mean()
    print(f"{dataname}:", mmd_loss)


def BMMD_score_naive(dataname, fake_data, length=24):
    if dataname == "energy":
        ori_data = np.load(f"./OUTPUT/samples/energy_norm_truth_{length}_train.npy")
    elif dataname == "stock":
        ori_data = np.load(f"./OUTPUT/samples/stock_norm_truth_{length}_train.npy")
    elif dataname == "sine":
        ori_data = np.load(f"./OUTPUT/samples/sine_ground_truth_{length}_train.npy")
    elif dataname == "fmri":
        ori_data = np.load(f"./OUTPUT/samples/fmri_norm_truth_{length}_train.npy")
    elif dataname == "mujoco":
        ori_data = np.load(f"./OUTPUT/samples/mujoco_norm_truth_{length}_train.npy")
    else:
        raise NotImplementedError(f"Unkown dataname: {dataname}")
    fake_data = unnormalize_to_zero_to_one(fake_data)
    fake_data = fake_data[: ori_data.shape[0]]
    ori_data = torch.tensor(ori_data).float()
    fake_data = torch.tensor(fake_data).float()

    ori_data = cross_correlation_distribution(ori_data).unsqueeze(-1)
    fake_data = cross_correlation_distribution(fake_data).unsqueeze(-1)

    assert ori_data.shape == fake_data.shape

    mmd_loss = BMMD_Naive(ori_data, fake_data, "rbf").mean()
    print(f"{dataname} FDDS Score:", mmd_loss)


def VDS_score(dataname, fake_data, length=24):
    if dataname == "energy":
        ori_data = np.load(f"./OUTPUT/samples/energy_norm_truth_{length}_train.npy")
    elif dataname == "stock":
        ori_data = np.load(f"./OUTPUT/samples/stock_norm_truth_{length}_train.npy")
    elif dataname == "sine":
        ori_data = np.load(f"./OUTPUT/samples/sine_ground_truth_{length}_train.npy")
    elif dataname == "fmri":
        ori_data = np.load(f"./OUTPUT/samples/fmri_norm_truth_{length}_train.npy")
    elif dataname == "mujoco":
        ori_data = np.load(f"./OUTPUT/samples/mujoco_norm_truth_{length}_train.npy")
    else:
        raise NotImplementedError(f"Unkown dataname: {dataname}")

    fake_data = unnormalize_to_zero_to_one(fake_data)
    fake_data = fake_data[: ori_data.shape[0]]
    ori_data = torch.tensor(ori_data).float()
    fake_data = torch.tensor(fake_data).float()

    vds_score = VDS_Naive(ori_data, fake_data, "rbf").mean()
    print(f"{dataname} VDS Score:", vds_score)
