# visualize_latent_space_pca.py

import os
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

def extract_latents(model, dataloader, device):
    """
    Extract latent mean vectors (mu) from the VAE encoder.

    Args:
        model (GeneExpressionVAE): Trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Device to run the model on ("cuda" or "cpu").
    
    Returns:
        torch.Tensor: Concatenated latent mean vectors.
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            encoded = model.encoder(batch)
            mu, _ = encoded.chunk(2, dim=-1)
            latents.append(mu.cpu())
    return torch.cat(latents, dim=0)


def plot_pca(latents, save_path=None):
    """
    Plot PCA projection of latent space (2D).

    Args:
        latents (numpy.ndarray): Latent vectors.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    
    Returns:
        None
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
    plt.title("PCA Projection of Latent Space")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Latent space plot saved to {save_path}")
    else:
        plt.show()


def visualize_latent_space(cfg):
    """
    Visualize the latent space of a trained VAE using PCA.

    Args:
        cfg (dict): Configuration dictionary containing paths and parameters.

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load input_dim from PCA model
    with open(cfg["pca_model_path"], "rb") as f:
        pca_model = pickle.load(f)
    input_dim = pca_model.n_components_

    model = GeneExpressionVAE(input_dim, cfg["hidden_dim"], cfg["latent_dim"]).to(device)
    model.load_state_dict(torch.load(cfg["weights_path"], map_location=device))
    dataloader = DataLoader(GeneExpressionDataset(cfg["data_path"]), batch_size=cfg["batch_size"], shuffle=False)

    latents = extract_latents(model, dataloader, device)
    plot_pca(latents.numpy(), save_path=os.path.join(cfg["output_dir"], f"latent_pca_{timestamp}.png"))


if __name__ == "__main__":
    from adVAE.config import gene_expression as cfg
    visualize_latent_space(cfg)
