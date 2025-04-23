import os
import pickle
import numpy as np
import torch
from datetime import datetime
from scipy.stats import ks_2samp, ttest_ind
from sklearn.metrics import mean_squared_error
from adVAE.visualization.plot_utils import plot_reconstruction_distribution

def inverse_transform(data_pca, pca_model, scaler):
    """
    Project PCA-reduced data back into original space.
    """
    return scaler.inverse_transform(pca_model.inverse_transform(data_pca))

def evaluate_synthetic_data(cfg, synthetic_path):
    """
    Evaluate synthetic gene expression data against real data using statistical metrics.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if cfg["data_path"].endswith(".npy"):
        real_pca = np.load(cfg["data_path"])
    elif cfg["data_path"].endswith(".pt"):
        real_pca = torch.load(cfg["data_path"])
        if isinstance(real_pca, torch.Tensor):
            real_pca = real_pca.numpy()
    else:
        raise ValueError("Unsupported file format for real PCA data.")

    with open(cfg["pca_model_path"], "rb") as f:
        pca_model = pickle.load(f)
    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)

    real = inverse_transform(real_pca, pca_model, scaler)

    if synthetic_path.endswith(".tsv"):
        synthetic = np.loadtxt(synthetic_path, delimiter="\t", skiprows=1, dtype=float)
    elif synthetic_path.endswith(".csv"):
        synthetic = np.loadtxt(synthetic_path, delimiter=",", skiprows=1, dtype=float)
    else:
        raise ValueError("Synthetic file must be .tsv or .csv")

    print(f"Real data shape: {real.shape} | Synthetic data shape: {synthetic.shape}")

    # Distribution-based comparisons
    ks_stat, ks_p = ks_2samp(real.flatten(), synthetic.flatten())
    t_stat, t_p = ttest_ind(real.flatten(), synthetic.flatten())

    real_mean = np.mean(real, axis=0)
    synthetic_mean = np.mean(synthetic, axis=0)
    mse = mean_squared_error(real_mean, synthetic_mean)
    mae = np.mean(np.abs(real_mean - synthetic_mean))

    # Compute Pearson correlation between mean vectors (only valid if same length)
    if real_mean.shape[0] == synthetic_mean.shape[0]:
        corr = np.corrcoef(real_mean, synthetic_mean)[0, 1]
    else:
        corr = float("nan")

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Pearson Correlation (mean vectors): {corr:.4f}")
    print(f"KS Test: p = {ks_p:.4g}, stat = {ks_stat:.4f}")
    print(f"T-Test: p = {t_p:.4g}, stat = {t_stat:.4f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    plot_path = os.path.join(cfg["output_dir"], f"synthetic_vs_real_distribution_{timestamp}.png")
    plot_reconstruction_distribution(real, synthetic, save_path=plot_path, title_prefix="Gene Expression")

if __name__ == "__main__":
    from adVAE.config import gene_expression as cfg
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d")

    synthetic_file = os.path.join(cfg["output_dir"], f"synthetic_gene_expression_{timestamp}.tsv")
    evaluate_synthetic_data(cfg, synthetic_file)