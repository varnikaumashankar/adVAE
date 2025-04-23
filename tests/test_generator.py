import unittest
import torch
from torch.utils.data import DataLoader
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.models.mri_vae import MRIVAE
from adVAE.utils.generator import generate_synthetic_data, generate_synthetic_data_from_posteriors
from adVAE.data_preprocessing.mri.dataset import MRIDataset
import os


class TestSyntheticGeneration(unittest.TestCase):

    def test_generate_gene_expression(self):
        model = GeneExpressionVAE(input_dim=13, hidden_dim=64, latent_dim=32)
        model.eval()
        samples = generate_synthetic_data(model, num_samples=5, latent_dim=32)
        self.assertEqual(samples.shape, (5, 13))

    def test_generate_mri_from_posteriors(self):
        dummy_data = torch.randn(8, 1, 128, 128)
        dummy_path = "tests/tmp_dummy_mri.pt"
        torch.save(dummy_data, dummy_path)

        dataset = MRIDataset(dummy_path)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = MRIVAE(latent_dim=64)
        model.eval()

        samples = generate_synthetic_data_from_posteriors(model, dataloader, num_samples=4, device="cpu")
        self.assertEqual(samples.shape, (4, 1, 128, 128))

        os.remove(dummy_path)

if __name__ == "__main__":
    unittest.main()
