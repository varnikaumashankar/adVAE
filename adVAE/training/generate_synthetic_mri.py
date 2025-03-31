import torch
import os
from PIL import Image
import numpy as np
from adVAE.models.mri_vae import MRIVAE
from adVAE.utils.generator import generate_synthetic_data


def save_single_slice_as_gif(slice_tensor, path):
    """
    Save a single grayscale image tensor (H, W) as a single-frame GIF.
    """
    array = slice_tensor.clamp(-1, 1)
    array = ((array + 1) / 2.0 * 255).byte().cpu().numpy()
    img = Image.fromarray(array, mode='L')
    img.save(path, save_all=False)


def generate_and_save(model_path, latent_dim, num_samples, output_dir, hidden_dim=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = MRIVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    synthetic = generate_synthetic_data(model, num_samples=num_samples, latent_dim=latent_dim, device=device)
    synthetic_tensor = torch.tensor(synthetic, dtype=torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    for i, img_tensor in enumerate(synthetic_tensor):
        for j, plane in enumerate(["tra", "sag", "cor"]):
            slice_tensor = img_tensor[j]  # (128, 128)
            save_path = os.path.join(output_dir, f"synthetic_mri_{i+1}_{plane}.gif")
            save_single_slice_as_gif(slice_tensor, save_path)

    print(f"Saved {num_samples * 3} synthetic MRI GIFs to {output_dir}")


if __name__ == "__main__":
    generate_and_save(
        model_path="results/mri/model/vae_weights.pth", latent_dim=64, num_samples=50,
        output_dir="results/mri/synthetic_mri_gifs")
