import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, log_var, reduction = 'sum'):
    """
    Computes the total VAE loss: reconstruction loss + KL divergence.

    Args:
        recon_x (Tensor): Reconstructed input.
        x (Tensor): Original input.
        mu (Tensor): Latent mean.
        log_var (Tensor): Latent log-variance.
        reduction (str): Reduction mode for reconstruction loss ('sum' or 'mean').

    Returns:
        Tensor: Combined VAE loss.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction = reduction)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div