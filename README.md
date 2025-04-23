# adVAE: Alzheimer's Data Variational Autoencoder

### The proposed Variational Autoencoder (VAE) model framework aims to aid with high-quality Alzheimer's Disease (AD) data augmentation.

Alzheimer's disease is a progressive neurodegenerative disease that primarily affects memory, thinking and other cognitive abilities. Its increasing incidence among the older population necessitates a need for more comprehensive research to understand its underlying mechanisms and develop effective diagnostic and therapeutic strategies.

Deep learning models are becoming increasingly popular in the analysis of transciptomic, multi-omic and other biomedical data aiding in research. However, a key challenge in training such models is their requirement for large datasets. Access to high-quality clinical data for Alzheimer’s research remains a challenge. While datasets like WashU’s OASIS and Harvard Medical School’s resources are publicly available, more comprehensive datasets such as ADNI come with significant access barriers - including strict data use agreements and institutional vetting. These requirements can limit participation from smaller research groups, underscoring the need for tools that can produce high-quality synthetic multimodal data to advance research when real-world data is scarce.

adVAE aims to tackle this challenge and aid Alzheimer's research by using Variational Autoencoders to synthesize artificial, high-quality multimodal data. This will be done by sampling from adVAE's modeled latent space.

Latent space refers to a lower-dimensional, compressed representation of complex, high-dimensional data. Instead of directly working with raw data, such as gene expression profiles or MRI images, adVAE encodes the data into a latent space. This space captures the underlying features of the data in a more compact form while retaining the key information needed to reconstruct the original data by sampling. While many Python libraries exist, adVAE will be using PyTorch's framework to create the VAE architecture and perform the latent space modeling.

<br>

**Input:** Biomedical data from AD individuals (Gene Expression (RNA Microarray), MRI).

**Output:** A latent space that is capable of generating more synthetic, high-quality multimodal data for AD research in the same format as that of the input data.

Each type of data will have its own dedicated model and pipeline, which means the data is processed and generated independently. This allows for tailored models for each type of data based on its specific characteristics.

<br>

**Programming language:** Python 3.10.14

**License:** OSI Approved - MIT License

<br>

### Installation:

Clone the repo and install dependencies by initialising conda environment.

```bash
git clone https://github.com/varnikaumashankar/adVAE.git
```

Alternatively, one can directly install the package without cloning the repo:

```bash
pip install dist/advae-0.1.0.tar.gz
```

<br>

### Initialisation:

**1. Create conda environment:** Ensure that your system has conda installed. On your terminal, run: `conda env create -f environment.yaml`  
**2. Activate conda environment:** Run: `conda activate adVAE`  
**3. Update dependencies in environment:** Run: `conda env update -f environment.yaml`

**Package Module Structure:** Check package's module structure by running `python adVAE/utils/utils.py`.

<br>

### Running the Pipeline:

Use `main.py` to run preprocessing, training, evaluation, or synthetic data generation for either pipeline.

```bash
python main.py --pipeline gene_expression --task preprocess
python main.py --pipeline gene_expression --task train
python main.py --pipeline gene_expression --task evaluate
python main.py --pipeline gene_expression --task visualize
python main.py --pipeline gene_expression --task generate
python main.py --pipeline gene_expression --task evaluate_synthetic

python main.py --pipeline mri --task preprocess
python main.py --pipeline mri --task train
python main.py --pipeline mri --task grid_search
python main.py --pipeline mri --task evaluate
python main.py --pipeline mri --task visualize
python main.py --pipeline mri --task generate
python main.py --pipeline mri --task evaluate_synthetic

```

All paths and parameters can be customized in `adVAE/config.py`.

<br>

### Tests:

**Test functionality by running unit-tests:**
1. Activate conda environment using `conda activate adVAE`  
2. Ensure data is stored in the correct location and `config.py` paths reflect it  
3. Move to the root directory and run unit tests with:

```bash
pytest tests/
```

### Tutorial:

**Example Dataset:**

A lightweight example dataset is included for quick testing:
1. Gene Expression: 5 `.tsv` files containing AMP-AD MSBB RNA microarray data from 5 subjects (~40,000 genes each)
2. MRI: 9 Gain-field corrected, atlas-registered average brain scans `.img` images × 20 slices from OASIS-1 


The example datasets are located in:
`data/example_gene_expression/`
`data/example_mri/`


A Jupyter notebook demonstrates the full pipeline on this example data:
`tutorials/example_data_tutorial.ipynb`

**Real Dataset:**

A full-scale real dataset is available for training and evaluation:
1. Gene Expression: 20 `.tsv` files containing AMP-AD MSBB RNA microarray data from 20 subjects covering ~40,000 genes across all subjects
2. MRI: ~100 Gain-field corrected, atlas-registered average brain scans `.img` images × 20 slices from OASIS-1 

Download the datasets from sources of choice or from sources mentioned in `Datasets.md` and move it into the `data/` folder.

A dedicated Jupyter notebook demonstrates the end-to-end training pipeline on the full dataset:
`tutorials/real_data_tutorial.ipynb`

<br>

### Note:

Due to limited data availability, the datasets were not split into the standard 70-20-10 training-validation-testing ratio as initially intended. As a result, model performance was evaluated directly on the training data, which may limit the generalizability of the results. Future iterations of this project will focus on incorporating proper validation and test splits to enable more rigorous performance assessment. This will be critical for building a more robust and generalizable VAE model capable of generating high-quality synthetic multimodal biomedical data. 