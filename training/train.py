import torch
from torch import optim
from torch import nn
from adVAE.models.vae import VAE
from adVAE.utils.data_loader import load_and_preprocess_data

def loss_function(recon_x, x, mu, log_var):
    """
    Calculates total VAE loss (sum of reconstruction loss and KL divergence)

    Args: 
        recon_x (torch.Tensor): The reconstructed data.
        x (torch.Tensor): The original input data.
        mu (torch.Tensor): The mean from the encoder's latent space.
        log_var (torch.Tensor): The logarithm of the variance from the encoder's latent space.

    Returns:
        reconstruction_loss: The total VAE loss. 
    """
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_loss

def train_model(data_path, input_dim, hidden_dim, latent_dim, epochs=10):
    """
    Trains a Variational Autoencoder (VAE) model.

    Args:
        data_path (str): Path to the training data.
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
        epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns:
        None
    """
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = load_and_preprocess_data(data_path)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch[0]
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss = loss_function(recon_batch, batch, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset):.4f}")
