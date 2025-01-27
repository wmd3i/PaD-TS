import torch
import argparse
import numpy as np
from resample import UniformSampler, Batch_Same_Sampler
from Model import PaD_TS
from diffmodel_init import create_gaussian_diffusion
from training import Trainer
from data_preprocessing.real_dataloader import CustomDataset
from data_preprocessing.sine_dataloader import SineDataset
from data_preprocessing.real_dataloader import fMRIDataset
from data_preprocessing.mujoco_dataloader import MuJoCoDataset
from torchsummary import summary
from data_preprocessing.sampling import sampling
from eval_run import (
    discriminative_score,
    predictive_score,
    BMMD_score,
    BMMD_score_naive,
    VDS_score,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data",
        "-d",
        default="energy",
        help="Data Name: [energy, stock, sine]",
        required=True,
    )
    parser.add_argument(
        "-window", "-w", default=24, type=int, help="Window Size", required=False
    )
    parser.add_argument(
        "-steps", "-s", default=0, type=int, help="Training Step", required=False
    )
    args = parser.parse_args()

    if args.data == "energy":
        from configs.energy_config import (
            Training_args,
            Model_args,
            Diffusion_args,
            DataLoader_args,
            Data_args,
        )
    elif args.data == "stock":
        from configs.stock_config import (
            Training_args,
            Model_args,
            Diffusion_args,
            DataLoader_args,
            Data_args,
        )
    elif args.data == "sine":
        from configs.sine_config import (
            Training_args,
            Model_args,
            Diffusion_args,
            DataLoader_args,
            Data_args,
        )
    elif args.data == "mujoco":
        from configs.mujoco_config import (
            Training_args,
            Model_args,
            Diffusion_args,
            DataLoader_args,
            Data_args,
        )
    elif args.data == "fmri":
        from configs.fmri_config import (
            Training_args,
            Model_args,
            Diffusion_args,
            DataLoader_args,
            Data_args,
        )
    else:
        raise NotImplementedError(f"Unkown Dataset: {args.data}")

    # Obtain the args from the config files
    train_arg = Training_args()
    model_arg = Model_args()
    diff_arg = Diffusion_args()
    dl_arg = DataLoader_args()
    d_arg = Data_args()

    if args.window != 24:
        d_arg.window = int(args.window)
        train_arg.save_dir = f"./OUTPUT/{d_arg.name}_{d_arg.window}_MMD/"
        model_arg.input_shape = (d_arg.window, d_arg.dim)

    if args.steps != 0:
        train_arg.lr_anneal_steps = args.steps

    print("======Load Data======")
    if d_arg.name == "sine":
        dataset = SineDataset(
            window=d_arg.window,
            num=d_arg.num,
            dim=d_arg.dim,
            save2npy=d_arg.save2npy,
            neg_one_to_one=d_arg.neg_one_to_one,
            seed=d_arg.seed,
            period=d_arg.period,
        )
    elif d_arg.name == "fmri":
        dataset = fMRIDataset(
            name=d_arg.name,
            proportion=d_arg.proportion,
            data_root=d_arg.data_root,
            window=d_arg.window,
            save2npy=d_arg.save2npy,
            neg_one_to_one=d_arg.neg_one_to_one,
            seed=d_arg.seed,
            period=d_arg.period,
        )
    elif d_arg.name == "mujoco":
        dataset = MuJoCoDataset(
            name=d_arg.name,
            window=d_arg.window,
            num=d_arg.num,
            dim=d_arg.dim,
            save2npy=d_arg.save2npy,
            neg_one_to_one=d_arg.neg_one_to_one,
            seed=d_arg.seed,
            period=d_arg.period,
        )
    else:
        dataset = CustomDataset(
            name=d_arg.name,
            proportion=d_arg.proportion,
            data_root=d_arg.data_root,
            window=d_arg.window,
            save2npy=d_arg.save2npy,
            neg_one_to_one=d_arg.neg_one_to_one,
            seed=d_arg.seed,
            period=d_arg.period,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dl_arg.batch_size,
        shuffle=dl_arg.shuffle,
        num_workers=dl_arg.num_workers,
        drop_last=dl_arg.drop_last,
        pin_memory=dl_arg.pin_memory,
    )

    # Create the model, diffusion, and schedule sampler
    model = PaD_TS(
        hidden_size=model_arg.hidden_size,
        num_heads=model_arg.num_heads,
        n_encoder=model_arg.n_encoder,
        n_decoder=model_arg.n_decoder,
        feature_last=model_arg.feature_last,
        mlp_ratio=model_arg.mlp_ratio,
        input_shape=model_arg.input_shape,
    )
    diffusion = create_gaussian_diffusion(
        predict_xstart=diff_arg.predict_xstart,
        diffusion_steps=diff_arg.diffusion_steps,
        noise_schedule=diff_arg.noise_schedule,
        loss=diff_arg.loss,
        rescale_timesteps=diff_arg.rescale_timesteps,
    )
    if train_arg.schedule_sampler == "batch":
        schedule_sampler = Batch_Same_Sampler(diffusion)
    elif train_arg.schedule_sampler == "uniform":
        schedule_sampler = UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"Unkown sampler: {train_arg.schedule_sampler}")

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=dl_arg.batch_size,
        lr=train_arg.lr,
        weight_decay=train_arg.weight_decay,
        lr_anneal_steps=train_arg.lr_anneal_steps,
        log_interval=train_arg.log_interval,
        save_interval=train_arg.save_interval,
        save_dir=train_arg.save_dir,
        schedule_sampler=schedule_sampler,
        mmd_alpha=train_arg.mmd_alpha,
    )
    summary(model)
    print("Loss Function: ", diff_arg.loss)
    print("Save Directory: ", train_arg.save_dir)
    print("Schedule Sampler: ", train_arg.schedule_sampler)
    print("Batch Size: ", dl_arg.batch_size)
    print("Diffusion Steps: ", diff_arg.diffusion_steps)
    print("Epochs: ", train_arg.lr_anneal_steps)
    print("Alpha: ", train_arg.mmd_alpha)
    print("Window Size: ", d_arg.window)
    print("Data shape: ", model_arg.input_shape)
    print("Hidden: ", model_arg.hidden_size)

    print("======Training======")
    trainer.train()
    print("======Done======")

    print("======Generate Samples======")
    concatenated_tensor = sampling(
        model,
        diffusion,
        dataset.sample_num,
        dataset.window,
        dataset.var_num,
        dl_arg.batch_size,
    )
    np.save(
        f"{train_arg.save_dir}ddpm_fake_{d_arg.name}_{dataset.window}.npy",
        concatenated_tensor.cpu(),
    )
    print(f"{train_arg.save_dir}ddpm_fake_{d_arg.name}_{dataset.window}.npy")

    print("======Diff Eval======")
    np_fake = np.array(concatenated_tensor.detach().cpu())
    print("======Discriminative Score======")
    discriminative_score(d_arg.name, 5, np_fake, length=d_arg.window)
    print("======Predictive Score======")
    predictive_score(d_arg.name, 5, np_fake, length=d_arg.window)
    # BMMD_score(d_arg.name, concatenated_tensor)
    print("======VDS Score======")
    VDS_score(d_arg.name, concatenated_tensor, length=d_arg.window)
    print("======FDDS Score======")
    BMMD_score_naive(d_arg.name, concatenated_tensor, length=d_arg.window)
    print("======Finished======")
