import torch
from tqdm import tqdm


def sampling(model, diffusion, sample_n, length, feature_n, batch_size, use_ddim=False):
    """Generate samples given the model."""
    model.eval()
    with torch.no_grad():
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        res = []
        for i in tqdm(range((sample_n // batch_size) + 1)):
            res.append(
                sample_fn(
                    model,
                    (batch_size, length, feature_n),
                    clip_denoised=True,
                )
            )
    concatenated_tensor = torch.cat(res, dim=0)
    return concatenated_tensor
