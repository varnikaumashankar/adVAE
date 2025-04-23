import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from scipy.stats import ks_2samp, ttest_ind
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from adVAE.models.mri_vae import MRIVAE
from adVAE.utils.generator import generate_synthetic_data_from_posteriors
from adVAE.visualization.plot_utils import plot_reconstruction_distribution

def evaluate_synthetic_data(cfg, synthetic_tensor):
    """
    Evaluate synthetic MRI data against real data using statistical metrics.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    real_dataset = MRIDataset(cfg["data_path"])
    real_loader = DataLoader(real_dataset, batch_size=len(real_dataset), shuffle=False)
    real_tensor = next(iter(real_loader)).squeeze(1).to(device)
    synthetic_tensor = synthetic_tensor.squeeze(1).to(device)

    real_np = real_tensor.cpu().numpy()
    synthetic_np = synthetic_tensor.cpu().numpy()

    print(f"Real data shape: {real_np.shape} | Synthetic data shape: {synthetic_np.shape}")

    # Distribution-level comparisons
    ks_stat, ks_p = ks_2samp(real_np.flatten(), synthetic_np.flatten())
    t_stat, t_p = ttest_ind(real_np.flatten(), synthetic_np.flatten())

    # Mean image-wise errors
    real_mean = np.mean(real_np, axis=0)
    synthetic_mean = np.mean(synthetic_np, axis=0)
    mse = np.mean((real_mean - synthetic_mean) ** 2)
    mae = np.mean(np.abs(real_mean - synthetic_mean))
    rmse = np.sqrt(mse)

    # Image quality metrics
    min_samples = min(len(real_np), len(synthetic_np))
    psnr_vals = [psnr(r, s, data_range=1.0) for r, s in zip(real_np[:min_samples], synthetic_np[:min_samples])]
    ssim_vals = [ssim(r, s, data_range=1.0) for r, s in zip(real_np[:min_samples], synthetic_np[:min_samples])]

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean PSNR: {np.mean(psnr_vals):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_vals):.4f}")
    print(f"KS Test: p = {ks_p:.4g}, stat = {ks_stat:.4f}")
    print(f"T-Test: p = {t_p:.4g}, stat = {t_stat:.4f}")

    plot_path = os.path.join(cfg["output_dir"], f"synthetic_vs_real_distribution_{timestamp}.png")
    plot_reconstruction_distribution(real_tensor[:min_samples], synthetic_tensor[:min_samples], save_path=plot_path, title_prefix="MRI")

if __name__ == "__main__":
    from adVAE.config import mri as cfg
    from torch.utils.data import DataLoader
    from adVAE.data_preprocessing.mri.dataset import MRIDataset
    from adVAE.training.generate_synthetic_mri import generate_synthetic_data_from_posteriors
    from adVAE.models.mri_vae import MRIVAE
    from adVAE.training.evaluate_synthetic_mri import evaluate_synthetic_data as evaluate_synthetic_mri_data


    model = MRIVAE(latent_dim=cfg["latent_dim"]) # Load model
    model.load_state_dict(torch.load(cfg["weights_path"], map_location="cpu"))
    model.eval()

    dataset = MRIDataset(cfg["data_path"]) # Load real data
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 32), shuffle=True)

    synthetic_np = generate_synthetic_data_from_posteriors( # Generate synthetic data using learned posterior
        model,
        dataloader=loader,
        num_samples=cfg["num_samples"],
        device="cpu"
    )

    synthetic_tensor = torch.tensor(synthetic_np, dtype=torch.float32) # Convert and evaluate
    evaluate_synthetic_mri_data(cfg, synthetic_tensor)
