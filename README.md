# adVAE: Alzheimer's Data Variational Autoencoder

### The proposed Variational Autoencoder (VAE) model framework aims to aid with high-quality Alzheimer's Disease (AD) data augmentation.

Alzheimer's disease is a progressive neurodegenerative disease that primarily affects memory, thinking and other cognitive abilities. Its increasing incidence among the older population necessitates a need for more comprehensive research to understand its underlying mechanisms and develop effective diagnostic and therapeutic strategies.

Deep learning models are becoming increasingly popular in the analysis of transciptomic, multi-omic and other biomedical data aiding in research. However, a key challenge in training such models is their requirement for large datasets. High-quality clinical data for Alzheimer's patients is scarce, and hence, data augmentation is needed to address the challenges posed by small sample sizes.

adVAE aims to tackle this challenge and aid Alzheimer's research by using Variational Autoencoders to synthesize artificial, high-quality multimodal data. This will be done by sampling from adVAE's modeled latent space.

Latent space refers to a lower-dimensional, compressed representation of complex, high-dimensional data. Instead of directly working with raw data, such as gene expression profiles or MRI images, adVAE encodes the data into a latent space. This space captures the underlying features of the data in a more compact form while retaining the key information needed to reconstruct the original data by sampling. While many Python libraries exist, adVAE will be using PyTorch's framework to create the VAE architecture and perform the latent space modeling.

<br>

**Input:** Biomedical data from AD individuals (gene expression, MRI).

**Output:** A latent space that is capable of generating more artificial, high-quality multimodal data for AD research.

Each type of data will have its own dedicated model and pipeline, which means the data is processed and generated independently. This allows for tailored models for each type of data based on its specific characteristics.

<br>
<br>

**Programming language:** Python 3.10

**License:** OSI Approved - MIT License

**Environment:** Recommended to have Great Lakes HPC access for GPU usage

<br>
<br>

### Initialization:

**1. Create conda environment:** Ensure that your system has conda installed. On your terminal, run: `conda env create -f environment.yaml`  
**2. Activate conda environment:** Run: `conda activate adVAE`  
**3. Update dependencies in environment:** Run: `conda env update -f environment.yaml`

<br>

### Running the Pipeline:

Use `main.py` to run preprocessing, training, evaluation, or synthetic data generation for either pipeline.

```bash
python main.py --pipeline gene_expression --task preprocess
python main.py --pipeline gene_expression --task train
python main.py --pipeline gene_expression --task evaluate
python main.py --pipeline gene_expression --task visualize
python main.py --pipeline gene_expression --task generate

python main.py --pipeline mri --task preprocess
python main.py --pipeline mri --task train
python main.py --pipeline mri --task evaluate
python main.py --pipeline mri --task visualize
python main.py --pipeline mri --task generate
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
1. Gene Expression: 5 `.tsv` files (~40,000 genes each)
2. MRI: 5 subjects × 3 slices (sag, cor, tra) → [3, 128, 128] tensors


The example datasets are located in:
`data/example_gene_expression/`
`data/example_mri/`


A Jupyter notebook demonstrates the full pipeline on this example data:
`tutorials/example_data_tutorial.ipynb`
