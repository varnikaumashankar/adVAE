import unittest
import pandas as pd
import torch
import os
from adVAE.data_preprocessing.gene_expression.preprocess import preprocess_pipeline as preprocess_gene
from adVAE.data_preprocessing.mri.preprocess import preprocess_pipeline as preprocess_mri

class TestPreprocessing(unittest.TestCase):

    def test_gene_expression_preprocessing(self):
        path = "data/AMP_AD_MSBB_MSSM"
        df, _, _, _ = preprocess_gene(
            n_components=0.95,
            data_folder=path,
            scale_method="standard",
            visualize=False,
            aggregate_by_gene=True
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[1], 0)
        self.assertFalse(df.isnull().values.any())

    def test_mri_preprocessing(self):
        preprocess_mri()

        processed_dir = "data/processed"
        matching_files = [f for f in os.listdir(processed_dir) if f.startswith("mri_") and f.endswith(".pt")]
        self.assertTrue(len(matching_files) > 0, "No timestamped MRI .pt file found in data/processed")

        path = os.path.join(processed_dir, matching_files[0])
        data = torch.load(path)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in data))
        self.assertTrue(all(t.shape == torch.Size([1, 128, 128]) for t in data))


if __name__ == "__main__":
    unittest.main()




    
    