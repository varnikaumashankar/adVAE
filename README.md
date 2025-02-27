# adVAE: Alzheimer's Data Variational Autoencoder

### The proposed Variational Autoencoder (VAE) model framework aims to aid with high-quality Alzheimer's Disease (AD) data augmentation.

Alzheimer's disease is a progressive neurodegenerative disease that primarily affects memory, thinking and other cognitive abilities. Its increasing incidence among the older population necessitates a need for more comprehensive research to understand its underlying mechanisms and develop effective diagnostic and therapeutic strategies.


Deep learning models are becoming increasingly popular in the analysis of transciptomic, multi-omic and other biomedical data aiding in research. However, a key challenge in training such models is their requirement for large datasets. High-quality clinical data for Alzheimer's patients is scarce, and hence, data augmentation is needed to address the challenges posed by small sample sizes.


adVAE aims to tackle this challenge and aid Alzheimer's research by using Variational Autoencoders to synthesize artificial, high-quality multimodal. This will be done by sampling from adVAE's modeled latent space.


Latent space refers to a lower-dimensional, compressed representation of complex, high-dimensional data. Instead of directly working with raw data, such as gene expression profiles, EEG signals, or MRI images, adVAE encodes the data into a latent space. This space captures the underlying features of the data in a more compact form while retaining the key information needed to reconstruct the original data by sampling. While many Python libraries exist, adVAE will be using PyTorch's framework to create the VAE architecture and perform the latent space modeling.

<br>

**Input:** Biomedical data from AD individuals (gene expression, EEG, MRI).

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

**1. Create conda environment:** Ensure that your system has conda installed. On your terminal, run: `conda env create -f environment.yaml`.
**2. Activate conda environment:** Run: `conda activate adVAE`.
**3. Update dependencies in environment:** Run `conda env update -f environment.yaml`.

### Tests:

**Test functionality by running unit-tests:**
1. Activate conda environment using `conda activate adVAE`.
2. Ensure data is stored in <data_folder>, and that the path to it has been edited in adVAE/data_preprocessing/gene_exp_preprocess.py and tests/test_training.py
3. Move to the root directory and run unit tests on the command line using `pytest tests/`.




