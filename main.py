import argparse
import os
import torch
from adVAE.config import gene_expression, mri
from adVAE.training.train_gene_expression import train_model as train_gene
from adVAE.training.train_mri import train_model as train_mri, grid_search as grid_search_mri
from adVAE.training.evaluate_gene_expression import evaluate_model as evaluate_gene
from adVAE.training.evaluate_mri import evaluate_model as evaluate_mri
from adVAE.visualization.visualize_gene_expression import visualize_latent_space as visualize_gene
from adVAE.visualization.visualize_mri import visualize_latent_space as visualize_mri
from adVAE.training.generate_synthetic_gene_expression import generate_and_save as generate_gene
from adVAE.training.generate_synthetic_mri import generate_and_save as generate_mri
from adVAE.data_preprocessing.gene_expression.preprocess import preprocess_pipeline as preprocess_gene
from adVAE.data_preprocessing.mri.preprocess import preprocess_pipeline as preprocess_mri
from adVAE.training.evaluate_synthetic_gene_expression import evaluate_synthetic_data as evaluate_synthetic_gene_data
from adVAE.training.evaluate_synthetic_mri import evaluate_synthetic_data as evaluate_synthetic_mri_data
from torch.utils.data import DataLoader
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from adVAE.training.generate_synthetic_mri import generate_synthetic_data_from_posteriors
from adVAE.models.mri_vae import MRIVAE
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    parser = argparse.ArgumentParser(description="Run adVAE pipeline components")
    parser.add_argument("--pipeline", choices=["gene_expression", "mri"], required=True)
    parser.add_argument("--task", choices=["preprocess", "train", "evaluate", "visualize", "generate", "grid_search", "evaluate_synthetic"], required=True)
    args = parser.parse_args()

    if args.pipeline == "gene_expression":
        cfg = gene_expression
        if args.task == "preprocess":
            print("Preprocessing gene expression data...")
            preprocess_gene(
                n_components=cfg["n_components"],
                data_folder=cfg["data_folder"],
                scale_method=cfg["scale_method"],
                visualize=cfg["visualize"],
                aggregate_by_gene=cfg["aggregate_by_gene"]
            )
            print("Preprocessing complete.")

        elif args.task == "train":
            print("Training GeneExpressionVAE on PCA data...")
            train_gene(cfg)
        elif args.task == "evaluate":
            print("Evaluating GeneExpressionVAE reconstruction...")
            evaluate_gene(cfg)
        elif args.task == "visualize":
            print("Visualizing GeneExpressionVAE latent space...")
            visualize_gene(cfg)
        elif args.task == "generate":
            print("Generating synthetic gene expression data...")
            generate_gene(cfg)
        elif args.task == "evaluate_synthetic":
            print("Evaluating synthetic gene expression data against real gene expression distribution...")
            synthetic_path = os.path.join(cfg["output_dir"], f"synthetic_gene_expression_{timestamp}.tsv")
            evaluate_synthetic_gene_data(cfg, synthetic_path)
        else:
            raise ValueError(f"Task '{args.task}' not implemented for pipeline '{args.pipeline}'")


    elif args.pipeline == "mri":
        cfg = mri
        if args.task == "train":
            print("Training MRIVAE on MRI data...")
            train_mri(data_path=cfg["data_path"], latent_dim=cfg["latent_dim"], batch_size=cfg["batch_size"], epochs=cfg["epochs"], lr=cfg["lr"], beta=cfg.get("beta", 1.0))
            print(f"Saved processed data to {cfg['output_dir']}")
        elif args.task == "evaluate":
            print("Evaluating performance of MRIVAE...")
            evaluate_mri(weights_path=cfg["weights_path"], data_path=cfg["data_path"], latent_dim=cfg["latent_dim"])
            print(f"Finished evaluating performance")
        elif args.task == "visualize":
            print("Visualizing latent space of MRIVAE...")
            visualize_mri(weights_path=cfg["weights_path"], data_path=cfg["data_path"], latent_dim=cfg["latent_dim"])
            print(f"Finished visualizing the latent space")
        elif args.task == "generate":
            print("Generating synthetic MRI data...")
            generate_mri(weights_path=cfg["weights_path"], latent_dim=cfg["latent_dim"], num_samples=cfg["num_samples"], output_dir=os.path.join(cfg["output_dir"], "synthetic_gifs"))
            print(f"Finished generating synthetic data")
        elif args.task == "preprocess":
            print("Preprocessing MRI data...")
            preprocess_mri()
            print(f"Saved processed MRI data to {cfg['data_path']}")
        elif args.task == "grid_search":
            print("Running grid search for MRIVAE...")
            grid_search_mri()
            print(f"Finished grid search")
        elif args.task == "evaluate_synthetic":
            print("Evaluating synthetic MRI data...")
            model = MRIVAE(latent_dim=cfg["latent_dim"])
            model.load_state_dict(torch.load(cfg["weights_path"], map_location="cpu"))
            model.eval()
            dataset = MRIDataset(cfg["data_path"])
            loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
            synthetic_np = generate_synthetic_data_from_posteriors(model, dataloader=loader, num_samples=cfg["num_samples"], device="cpu")
            synthetic_tensor = torch.tensor(synthetic_np, dtype=torch.float32)
            evaluate_synthetic_mri_data(cfg, synthetic_tensor)
        else:
            raise ValueError(f"Task '{args.task}' not implemented for pipeline '{args.pipeline}'")


if __name__ == "__main__":
    main()
