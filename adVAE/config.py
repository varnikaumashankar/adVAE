from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

gene_expression = {"hidden_dim": 32, "latent_dim": 8, "batch_size": 4, "epochs": 100, "lr": 0.001, "num_samples": 20, "data_folder": "data/AMP_AD_MSBB_MSSM", "n_components": 0.95, "scale_method": "standard", "visualize": False, "aggregate_by_gene": True, "num_samples": 10,
                   "data_path": f"data/processed/X_pca_{timestamp}.npy", "weights_path": f"results/gene_expression/model/vae_weights_{timestamp}.pth", "output_dir": "results/gene_expression", "beta": 0.1, "pca_model_path": f"data/processed/pca_model_{timestamp}.pkl", "scaler_path": f"data/processed/scaler_{timestamp}.pkl"}

mri = {"latent_dim": 64, "batch_size": 32, "epochs": 50, "lr": 0.001, "data_folder": "data/OASIS_1", "data_path": f"data/processed/mri_{timestamp}.pt",
       "weights_path": f"results/mri/vae_weights_lat64_b0.1_lr0.001_{timestamp}.pth", "output_dir": "results/mri", "beta": 0.1, "num_samples": 10}
