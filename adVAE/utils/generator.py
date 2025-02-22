import torch

def generate_synthetic_data(model, num_samples, latent_dim):
    """
    Generates synthetic gene expression data using a trained generative model.

    Args:
        model (torch.nn.Module): The trained generative model with a decoder method.
        num_samples (int): The number of synthetic data samples to generate.
        latent_dim (int): The dimensionality of the latent space.

    Returns:
        numpy.ndarray: The generated synthetic data as a NumPy array.
    """

    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        generated_data = model.decoder(z)
    return generated_data.cpu().numpy()