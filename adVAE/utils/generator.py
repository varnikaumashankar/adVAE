import torch

def generate_synthetic_data(model, num_samples, latent_dim, device="cpu"):
    """
    Generates synthetic gene expression data using the decoder of a trained VAE.

    Args:
        model (torch.nn.Module): The trained generative model with a decoder.
        num_samples (int): Number of synthetic samples to generate.
        latent_dim (int): Dimension of the latent space.
        device (str): Device to run the generation on.

    Returns:
        numpy.ndarray: Generated synthetic data as a NumPy array.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        if hasattr(model, 'decoder_input'):
            x = model.decoder_input(z).view(-1, 128, 16, 16)
            generated_data = model.decoder(x)
        else:
            generated_data = model.decoder(z)
    return generated_data.cpu().numpy()