import torch
from torch import nn

class GeneExpressionVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for PCA-reduced gene expression data.

    Args:
        input_dim (int): Dimensionality of input features (e.g., 100 PCs).
        hidden_dim (int): Dimensionality of hidden layers.
        latent_dim (int): Dimensionality of latent space.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GeneExpressionVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) 
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).

        Args:
            mu (Tensor): Mean of latent distribution.
            log_var (Tensor): Log variance of latent distribution.

        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the encoder, reparameterization, and decoder.

        Args:
            x (Tensor): Input batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Reconstructed input, mu, log_var
        """
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
