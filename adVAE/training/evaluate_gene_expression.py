import os
import torch
from torch.utils.data import DataLoader
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from adVAE.metrics.performance import reconstruction_accuracy, mean_absolute_error
from adVAE.visualization.plot_utils import plot_reconstruction_distribution


def evaluate_model(model_path, data_path, input_dim, hidden_dim=64, latent_dim=32, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu",
                    save_dir="results/gene_expression"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = GeneExpressionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = GeneExpressionVAE(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_original = []
    all_reconstructed = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
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
    evaluate_model(
        model_path="results/gene_expression/model/vae_weights.pth",
        data_path="data/processed/gene_expression.pt",
        input_dim=13
    )