gene_expression = {"input_dim": 13, "hidden_dim": 64, "latent_dim": 32, "batch_size": 64, "epochs": 50, "lr": 0.001, "data_folder": "data/AMP_AD_MSBB_MSSM", 
                   "data_path": "data/processed/gene_expression.pt", "weights_path": "results/gene_expression/model/vae_weights.pth", "results_dir": "results/gene_expression"}

mri = {"latent_dim": 64, "batch_size": 32, "epochs": 30, "lr": 0.001, "data_folder": "data/OASIS_1", "data_path": "data/processed/mri.pt",
       "weights_path": "results/mri/model/vae_weights.pth", "results_dir": "results/mri"}
