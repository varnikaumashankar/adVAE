import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, log_var, beta=0.5, reduction='sum'):
    """
    Compute the VAE loss function (reconstruction + beta * KL divergence).

    Args:
        recon_x (torch.Tensor): Reconstructed image tensor.
        x (torch.Tensor): Original image tensor.
        mu (torch.Tensor): Mean of the latent space.
        log_var (torch.Tensor): Log variance of the latent space.
        beta (float): Weight for the KL divergence term.
        reduction (str): Reduction method ('sum' or 'mean').
    
    Returns:
        total_loss (torch.Tensor): Total loss value.  
    """
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)

    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_div
    return total_loss
