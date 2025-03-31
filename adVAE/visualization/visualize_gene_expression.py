import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from torch.utils.data import DataLoader


def extract_latents(model, dataloader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            encoded = model.encoder(batch)
            mu, _ = encoded.chunk(2, dim=-1)
            latents.append(mu.cpu())
    return torch.cat(latents, dim=0)


def plot_pca(latents, save_path=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
    plt.title("PCA Projection of Latent Space")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Latent space plot saved to {save_path}")
    else:
        plt.show()


def visualize_latent_space(model_path, data_path, input_dim, hidden_dim=64, latent_dim=32, batch_size=64,
                            device="cuda" if torch.cuda.is_available() else "cpu", save_dir="results/gene_expression"):
    os.makedirs(save_dir, exist_ok=True)
    dataset = GeneExpressionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = GeneExpressionVAE(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    latents = extract_latents(model, dataloader, device)
    plot_pca(latents.numpy(), save_path=os.path.join(save_dir, "latent_pca.png"))


if __name__ == "__main__":
    visualize_latent_space(model_path="results/gene_expression/model/vae_weights.pth", data_path="data/processed/gene_expression.pt",
                           input_dim=13, hidden_dim=64, latent_dim=32)
