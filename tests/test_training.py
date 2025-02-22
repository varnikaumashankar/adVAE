import unittest
from training.train import train_model

class TestTraining(unittest.TestCase):
    def test_training_process(self): # Edit test data path
        train_model("/Users/varnikaumashankar/Documents/UMich/Semester 2/BIOINF 576/adVAE/data/AMP_AD_MSBB_MSSM/AMP-AD_MSBB_MSSM_AffymetrixU133AB_Frontal Pole_RSPP-adj_tommodules.tsv", input_dim=13, hidden_dim=64, latent_dim=10, epochs=1)