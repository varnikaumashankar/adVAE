import os
import torch
from torch.utils.data import DataLoader
from adVAE.models.mri_vae import MRIVAE
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from adVAE.metrics.performance import reconstruction_accuracy, mean_absolute_error
from adVAE.visualization.plot_utils import plot_reconstruction_distribution


def evaluate_model(model_path, data_path, latent_dim=64, batch_size=32, save_dir="results/mri",
    device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = MRIDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MRIVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_original = []
    all_reconstructed = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon, _, _ = model(batch)
            all_original.append(batch.cpu())
            all_reconstructed.append(recon.cpu())

    original = torch.cat(all_original, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)

    acc = reconstruction_accuracy(original, reconstructed, threshold=0.1)
    mae = mean_absolute_error(original, reconstructed)

    print(f"Reconstruction Accuracy: {acc:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    plot_reconstruction_distribution(original, reconstructed, save_path=os.path.join(save_dir, "reconstruction_distribution.png"))


if __name__ == "__main__":
    evaluate_model(model_path="results/mri/model/vae_weights.pth", data_path="data/processed/mri.pt",
        latent_dim=64, batch_size=32)