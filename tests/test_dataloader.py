import unittest
import torch
from datetime import datetime
from adVAE.utils.data_loader import load_gene_expression_data, load_mri_data

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
        self.gene_data_path = f"data/processed/X_pca_{self.timestamp}.npy"
        self.mri_data_path = f"data/processed/mri_{self.timestamp}.pt"

    def test_load_gene_expression_data(self):
        batch_size = 4
        dataloader = load_gene_expression_data(self.gene_data_path, batch_size=batch_size)
        batch = next(iter(dataloader))

        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape[0], batch_size)
        self.assertGreater(batch.shape[1], 0) 

    def test_load_mri_data(self):
        batch_size = 2
        dataloader = load_mri_data(self.mri_data_path, batch_size=batch_size)
        batch = next(iter(dataloader))

        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape[0], batch_size)
        self.assertEqual(batch.shape[1:], (1, 128, 128))  

if __name__ == '__main__':
    unittest.main()
