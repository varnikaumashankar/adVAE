import torch

def generate_synthetic_data(model, num_samples, latent_dim, device="cpu"):
    """
    Generate synthetic gene expression PCA data from standard normal latent space.

    Args:
        model (torch.nn.Module): Trained VAE model.
        num_samples (int): Number of samples to generate.
        latent_dim (int): Size of the latent space.
        device (str): Device to generate on ('cpu' or 'cuda').

    Returns:
        numpy.ndarray: Generated PCA-space samples.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_data = model.decoder(z)
    return generated_data.detach().cpu().numpy()

def generate_synthetic_data_from_posteriors(model, dataloader, num_samples, device="cpu"):
    """
    Generate synthetic MRI data by sampling z from the encoder's learned posterior (q(z|x)).

    Args:
        model (torch.nn.Module): The trained VAE model.
        dataloader (DataLoader): Dataloader with real MRI data.
        num_samples (int): Number of synthetic samples to generate.
        device (str): Device to run computations on.

    Returns:
        numpy.ndarray: Generated synthetic MRI images.
    """
    model.to(device)
    model.eval()
    zs = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, mu, log_var = model(batch)
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
            zs.append(z)
            if len(torch.cat(zs, dim=0)) >= num_samples:
                break

    z = torch.cat(zs, dim=0)[:num_samples]
    x = model.decoder_input(z).view(-1, 128, 16, 16)
    generated_data = model.decoder(x)
    return generated_data.detach().cpu().numpy()
