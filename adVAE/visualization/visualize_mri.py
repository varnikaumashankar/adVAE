import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from adVAE.models.mri_vae import MRIVAE
from adVAE.data_preprocessing.mri.dataset import MRIDataset


def extract_latents(model, dataloader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x = model.encoder(batch)
            mu, _ = model.fc_mu(x), model.fc_logvar(x)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Latent space plot saved to {save_path}")
    else:
        plt.show()


def visualize_latent_space(model_path, data_path, latent_dim=64, batch_size=32, 
                          save_dir="results/mri", device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(save_dir, exist_ok=True)
    dataset = MRIDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = MRIVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    latents = extract_latents(model, dataloader, device)
    plot_pca(latents.numpy(), save_path=os.path.join(save_dir, "latent_pca.png"))


if __name__ == "__main__":
    visualize_latent_space(model_path="results/mri/model/vae_weights.pth",
        data_path="data/processed/mri.pt", latent_dim=64)
