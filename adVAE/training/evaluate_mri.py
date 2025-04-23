import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind, ks_2samp
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error
from adVAE.models.mri_vae import MRIVAE
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from adVAE.metrics.performance import reconstruction_accuracy, mean_absolute_error
from adVAE.visualization.plot_utils import plot_reconstruction_distribution
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

def compute_metrics(img1, img2):
    """
    Compute various metrics between two images. (Peak Signal-to-Noise Ratio, Mean Squared Error, Root Mean Squared Error, Structural Similarity Index, Mean Absolute Error)

    Args:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
    
    Returns:
        tuple: A tuple containing SSIM, PSNR, MSE, RMSE, and MAE values.
    """
    i1 = img1.squeeze().numpy()
    i2 = img2.squeeze().numpy()
    ssim_val = ssim(i1, i2, data_range=1.0) # SSIM helps to measure the structural similarity between two images
    psnr_val = psnr(i1, i2, data_range=1.0) # PSNR is a measure of the peak error 
    mse_val = mean_squared_error(i1, i2) # MSE is the average of the squares of the errors
    rmse_val = np.sqrt(mse_val) # RMSE is the square root of the average of the squares of the errors
    mae_val = np.abs(i1 - i2).mean() # MAE is the average of the absolute differences between two images
    return ssim_val, psnr_val, mse_val, rmse_val, mae_val

def save_image_pair(original, reconstructed, output_path):
    """
    Save a pair of original and reconstructed images side by side.

    Args:
        original (torch.Tensor): Original image tensor.
        reconstructed (torch.Tensor): Reconstructed image tensor.
        output_path (str): Path to save the image.
    
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original.squeeze().numpy(), cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(reconstructed.squeeze().numpy(), cmap='gray')
    axes[1].set_title('Reconstruction')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def histogram_comparison(real, fake, output_path):
    """
    Compare the histograms of real and synthetic images.
    
    Args:
        real (torch.Tensor): Real image tensor.
        fake (torch.Tensor): Synthetic image tensor.
        output_path (str): Path to save the histogram comparison.
        
    Returns:
        None
    """
    real_vals = real.view(-1).numpy()
    fake_vals = fake.view(-1).numpy()
    plt.hist(real_vals, bins=100, alpha=0.5, label='Real', density=True)
    plt.hist(fake_vals, bins=100, alpha=0.5, label='Synthetic', density=True)
    plt.legend()
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.savefig(output_path)
    plt.close()

def evaluate_model(weights_path, data_path, latent_dim=64, batch_size=32, output_dir="results/mri", device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(output_dir, exist_ok=True)

    dataset = MRIDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MRIVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_original = []
    all_reconstructed = []
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    rmse_scores = []
    mae_scores = []

    csv_path = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "SSIM", "PSNR", "MSE", "RMSE", "MAE"])

        with torch.no_grad():
            idx = 0
            for batch in dataloader:
                batch = batch.to(device)
                recon, _, _ = model(batch)
                for i in range(len(batch)):
                    ssim_val, psnr_val, mse_val, rmse_val, mae_val = compute_metrics(batch[i].cpu(), recon[i].cpu())
                    ssim_scores.append(ssim_val)
                    psnr_scores.append(psnr_val)
                    mse_scores.append(mse_val)
                    rmse_scores.append(rmse_val)
                    mae_scores.append(mae_val)
                    writer.writerow([idx, ssim_val, psnr_val, mse_val, rmse_val, mae_val])
                    if idx < 10:
                        save_image_pair(batch[i].cpu(), recon[i].cpu(), os.path.join(output_dir, f"recon_{idx}_{timestamp}.png"))
                    idx += 1
                all_original.append(batch.cpu())
                all_reconstructed.append(recon.cpu())

    original = torch.cat(all_original, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)

    acc = reconstruction_accuracy(original, reconstructed, threshold=0.1)
    mae = mean_absolute_error(original, reconstructed)
    ssim_mean = np.mean(ssim_scores)
    psnr_mean = np.mean(psnr_scores)
    mse_mean = np.mean(mse_scores)
    rmse_mean = np.mean(rmse_scores)

    print(f"Reconstruction Accuracy: {acc:.4f}")
    print(f"Mean SSIM: {ssim_mean:.4f}")
    print(f"Mean PSNR: {psnr_mean:.4f}")
    print(f"Mean MSE: {mse_mean:.4f}")
    print(f"Mean RMSE: {rmse_mean:.4f}")
    print(f"Mean MAE: {mae:.4f}")


    histogram_comparison(original, reconstructed, output_path=os.path.join(output_dir, f"histogram_comparison_{timestamp}.png")) # compare histograms to compare the distributions of pixel values

    real_vals = original.view(-1).numpy()
    fake_vals = reconstructed.view(-1).numpy()

    t_stat, t_p = ttest_ind(real_vals, fake_vals) # perform t-test to compare the means of two independent samples 
    ks_stat, ks_p = ks_2samp(real_vals, fake_vals) # perform Kolmogorov-Smirnov test to compare the distributions of two independent samples

    print(f"t-test: stat = {t_stat:.4f}, p = {t_p:.4e}") 
    print(f"KS-test: stat = {ks_stat:.4f}, p = {ks_p:.4e}")

    plot_reconstruction_distribution(original, reconstructed, save_path=os.path.join(output_dir, f"reconstruction_distribution_{timestamp}.png")) # plot the distribution of pixel values in the original and reconstructed images

if __name__ == "__main__":
    evaluate_model(weights_path=f"results/mri/vae_weights_lat64_b0.1_lr0.001_{timestamp}.pth", data_path=f"data/processed/mri_{timestamp}.pt", latent_dim=64, batch_size=32)