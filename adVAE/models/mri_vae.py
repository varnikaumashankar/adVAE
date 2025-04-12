import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block for the VAE model.
    This block consists of two convolutional layers with batch normalization and ReLU activation.
    The output of the block is added to the input (skip connection) before applying ReLU activation.

    Args:
        channels (int): Number of input and output channels for the convolutional layers.

    Attributes:
        conv (nn.Sequential): Sequential container for the convolutional layers, batch normalization, and ReLU activation.
    
    Methods:
        forward(x): Forward pass through the block, applying the convolutional layers and adding the skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(self.conv(x) + x)

class MRIVAE(nn.Module):
    """
    Variational Autoencoder for MRI data.
    This model is designed to encode and decode MRI images using a convolutional architecture.
    Residual blocks are used in both encoder and decoder for better gradient flow and feature learning.

    Args:
        latent_dim (int): Dimensionality of the latent space.
    
    Attributes:
        latent_dim (int): Dimensionality of the latent space.
        encoder (nn.Sequential): Encoder network consisting of convolutional layers.
        fc_mu (nn.Linear): Fully connected layer to compute the mean of the latent space.
        fc_logvar (nn.Linear): Fully connected layer to compute the log variance of the latent space.
        decoder_input (nn.Linear): Fully connected layer to transform the latent space back to image dimensions.
        decoder (nn.Sequential): Decoder network consisting of transposed convolutional layers.

    Methods:
        reparameterize(mu, logvar): Reparameterization trick to sample from the latent space.
        forward(x): Forward pass through the network, returning reconstructed images and latent variables.
    """
    def __init__(self, latent_dim=64):
        super(MRIVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_input(z).view(-1, 128, 16, 16)
        x = self.decoder(x)
        return x, mu, logvar