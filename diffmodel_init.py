

import argparse
import inspect
import numpy as np
import torch as th

import gaussian_diffusion_loss as gd
from respace import space_timesteps, SpacedDiffusion


        

def diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
    )
    

def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    loss='MSE',
    predict_xstart=False,
    rescale_timesteps=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    assert loss in ['MSE', 'MSE_MMD', 'MSE_CORR']
    
    if loss == 'MSE_MMD':
        loss_type = gd.LossType.MSE_MMD
    elif loss == 'MSE_CORR':
        loss_type = gd.LossType.MSE_CORR
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

