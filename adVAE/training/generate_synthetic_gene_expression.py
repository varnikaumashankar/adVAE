import torch
import os
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.utils.generator import generate_synthetic_data
import pandas as pd

def generate_and_save(model_path, latent_dim, num_samples, output_path, input_dim, hidden_dim=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = GeneExpressionVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))

    synthetic = generate_synthetic_data(model, num_samples=num_samples, latent_dim=latent_dim, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(synthetic)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Generated {num_samples} synthetic samples saved to {output_path}")


if __name__ == "__main__":
    generate_and_save(model_path="results/gene_expression/model/vae_weights.pth", latent_dim=32,
        num_samples=100, output_path="results/gene_expression/synthetic_gene_expression.tsv", input_dim=13)