import unittest
import torch
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.models.mri_vae import MRIVAE
from adVAE.metrics.performance import reconstruction_accuracy, mean_absolute_error

class TestEvaluation(unittest.TestCase):
    def test_evaluate_gene_expression_model(self):
        model = GeneExpressionVAE(input_dim=13, hidden_dim=64, latent_dim=32)
        x = torch.randn(10, 13)
        with torch.no_grad():
            recon, _, _ = model(x)
        acc = reconstruction_accuracy(x, recon, threshold=0.1)
        mae = mean_absolute_error(x, recon)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
        self.assertGreaterEqual(mae, 0)

    def test_evaluate_mri_model(self):
        model = MRIVAE(latent_dim=64)
        x = torch.randn(4, 1, 128, 128) 
        with torch.no_grad():
            recon, _, _ = model(x)
        acc = reconstruction_accuracy(x, recon, threshold=0.1)
        mae = mean_absolute_error(x, recon)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
        self.assertGreaterEqual(mae, 0)

if __name__ == "__main__":
    unittest.main()
