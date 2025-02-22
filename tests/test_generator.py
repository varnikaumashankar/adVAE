import unittest
from adVAE.models.vae import VAE
from adVAE.utils.generator import generate_synthetic_data

class TestGenerator(unittest.TestCase):
    def test_data_generation(self):
        model = VAE(13, 64, 10)
        generated = generate_synthetic_data(model, 5, 10)
        self.assertEqual(generated.shape, (5, 13))