import torch
from torch import nn

class VAE(nn.Module): 
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize the VAE model.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterize the latent variables.

        Args:
            mu (Tensor): Mean of the latent Gaussian.
            log_var (Tensor): Log variance of the latent Gaussian.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Perform a forward pass through the VAE.

        Args:
            x (Tensor): Input tensor.
        """
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim = -1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

