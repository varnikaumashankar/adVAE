import torch
import os
import pickle
import pandas as pd
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.utils.generator import generate_synthetic_data
from sklearn.decomposition import PCA
from datetime import datetime


def generate_and_save(cfg):
    """
    Generate synthetic gene expression data using VAE in PCA space,
    then inverse transform to original gene space.
    
    Args:
        cfg (dict): Configuration dictionary
    
    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(cfg["pca_model_path"], "rb") as f:
        pca_model = pickle.load(f)
    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)

    input_dim = pca_model.n_components_

    model = GeneExpressionVAE(input_dim=input_dim, hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"]) # Load model and generate in latent space
    model.load_state_dict(torch.load(cfg["weights_path"], map_location=device))

    synthetic_pca = generate_synthetic_data(
        model, num_samples=cfg["num_samples"], latent_dim=cfg["latent_dim"], device=device
    )

    synthetic_scaled = pca_model.inverse_transform(synthetic_pca)  # Inverse transform: PCA → scaled gene → real gene
    synthetic_real = scaler.inverse_transform(synthetic_scaled)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    columns = ['k.all', 'k.in', 'k.out', 'k.diff', 'k.in.normed', 'k.all.normed', 'to.all',
        'to.in', 'to.out', 'to.diff', 'to.in.normed', 'to.all.normed', 'standard.deviation']
    df = pd.DataFrame(synthetic_real, columns=columns)
    out_path = os.path.join(cfg["output_dir"], f"synthetic_gene_expression_{timestamp}.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Generated {cfg['num_samples']} synthetic samples saved to {out_path}")

if __name__ == "__main__":
    from adVAE.config import gene_expression as cfg
    generate_and_save(cfg)

    