import os
import torch
import pickle
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from scipy.stats import ks_2samp, ttest_ind, f_oneway
from sklearn.metrics import mean_squared_error
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from adVAE.metrics.performance import reconstruction_accuracy, mean_absolute_error
from adVAE.visualization.plot_utils import plot_reconstruction_distribution
from sklearn.decomposition import PCA


def evaluate_model(cfg):
    """
    Evaluate VAE reconstruction on PCA-reduced gene expression data.
    Reconstructs back to original gene expression space.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    with open(cfg["pca_model_path"], "rb") as f:
        pca_model = pickle.load(f)
    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    input_dim = pca_model.n_components_

    dataset = GeneExpressionDataset(cfg["data_path"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = GeneExpressionVAE(input_dim, cfg["hidden_dim"], cfg["latent_dim"]).to(device)
    model.load_state_dict(torch.load(cfg["weights_path"], map_location=device))
    model.eval()

    all_original, all_reconstructed = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon, _, _ = model(batch)
            all_original.append(batch.cpu())
            all_reconstructed.append(recon.cpu())

    original_pca = torch.cat(all_original, dim=0).numpy()
    reconstructed_pca = torch.cat(all_reconstructed, dim=0).numpy()

    original = scaler.inverse_transform(pca_model.inverse_transform(original_pca)) 
    reconstructed = scaler.inverse_transform(pca_model.inverse_transform(reconstructed_pca))

    # Reconstruction metrics
    acc = reconstruction_accuracy(torch.tensor(original), torch.tensor(reconstructed), threshold=0.1)
    mae = mean_absolute_error(torch.tensor(original), torch.tensor(reconstructed))
    mse = mean_squared_error(original, reconstructed)
    print(f"Reconstruction Accuracy (original space): {acc:.4f}")
    print(f"Mean Absolute Error (original space): {mae:.4f}")
    print(f"Mean Squared Error (original space): {mse:.4f}")

    # Statistical tests
    ks_stat, ks_p = ks_2samp(original.flatten(), reconstructed.flatten())
    t_stat, t_p = ttest_ind(original.flatten(), reconstructed.flatten())
    print(f"KS Test p-value: {ks_p:.4g} | Statistic: {ks_stat:.4f}")
    print(f"T-Test p-value: {t_p:.4g} | Statistic: {t_stat:.4f}")

    # Optional: Pearson correlation per feature
    corr_per_feature = [np.corrcoef(original[:, i], reconstructed[:, i])[0, 1] for i in range(original.shape[1])]
    mean_corr = np.nanmean(corr_per_feature)
    print(f"Mean Pearson Correlation across features: {mean_corr:.4f}")

    plot_reconstruction_distribution(original, reconstructed,
        save_path=os.path.join(cfg["output_dir"], f"reconstruction_distribution_{timestamp}.png"))


if __name__ == "__main__":
    from adVAE.config import gene_expression as cfg
    evaluate_model(cfg)