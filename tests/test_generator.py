import unittest
import torch
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.models.mri_vae import MRIVAE
from adVAE.utils.generator import generate_synthetic_data


class TestSyntheticGeneration(unittest.TestCase):

    def test_generate_gene_expression(self):
        model = GeneExpressionVAE(input_dim=13, hidden_dim=64, latent_dim=32)
        model.eval()
        samples = generate_synthetic_data(model, num_samples=5, latent_dim=32)
        self.assertEqual(samples.shape, (5, 13))

    def test_generate_mri(self):
        model = MRIVAE(latent_dim=64)
        model.eval()
        samples = generate_synthetic_data(model, num_samples=4, latent_dim=64)
        self.assertEqual(samples.shape, (4, 3, 128, 128))