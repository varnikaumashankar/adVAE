# Dataset Information:

adVAE is designed to work with two types of Alzheimer’s-specific biomedical data: Gene expression profiles (RNA Microarray) and Gain-field corrected, atlas-registered average MRI brain scans. 

<br>

### Gene Expression Data:

**Input:** Gene expression profiles (RNA microarray data) from AD individuals.

**Data Format:**
- .tsv files

**Data Preprocessing:**
- Ensure all data is RNA microarray data obtained from human brain tissue of AD patients
- Remove duplicate values
- Fill missing values
- Aggregate by gene
- Scale (Standard/MinMax)
- PCA (retain 95% Variance)
- Ideal data split: 
    - 500 samples (example data - used for tool development)
    - Remaining samples (70-20-10 into training, validation & test data)

**Sources:**
AMP-AD MSSM Dataset from AD Knowledge Portal: https://adknowledgeportal.synapse.org/Explore/Programs/DetailsPage?Program=Resilience-AD

<br>

### MRI Data: 

**Input:** Magnetic Resonance Imaging data obtained from AD individuals.

**Data Format:**
- Transverse slices extracted from Gain-field corrected, atlas-registered average brain scans (.img)

**Data Preprocessing:**
- Ensure all data is MRI images of the brains obtained from humans who have AD
- Ensure proper orientation and extract 2D slices from 3D volumes
- Standardize resolution to 128x128 pixels
- Convert to grayscale tensors
- Normalize intensities to [-1, 1]
- Ideal data split: 
    - 100-400 images (example data - used for tool development)
    - Remaining samples (70-20-10 into training, validation & test data)

**Sources:**
OASIS-1 Dataset from WashU: https://sites.wustl.edu/oasisbrains/home/oasis-1/

<br>

### Example Data:

**Gene Expression Data:**
I use a subset of gene expression data consisting of 5 samples, each represented by a .tsv format. The dataset will be preprocessed by handling missing values, removing duplicate values, aggregating by gene, scaling the gene expression levels, and performing PCA to retain 95% variance as per the implemented pipeline. Basic stats: 5 samples, ~40,000 genes each.

This example dataset is ideal for running adVAE because:
- It is small enough (< 20MB) to be used during development and testing.
- It captures the key data structure used in real AD studies (transcriptomics data from human brain tissue).
- The dimensionality and complexity are sufficient to demonstrate the utility of a generative model like a VAE without requiring GPU acceleration.

The example dataset can be found in the `data/example_gene_expression/` directory in this repository.

**MRI Data:**
I use a subset of MRI data consisting of 9 Gain-field corrected, atlas-registered average brain scans images, each in .img format that was used to extract transverse slices. These images are preprocessed through the adVAE pipeline to ensure standardized orientation and size (128x128 Grayscale), and resolution normalization [-1, 1]. Basic stats: 9 subjects, 20 slices each.

This example dataset is ideal for running adVAE because:
- It is lightweight (< 20MB), making it suitable for rapid development and debugging.
- It reflects the typical structure of 3D neuroimaging data used in Alzheimer's research.
- It enables demonstration of image-based generative modeling using VAEs, even on resource-limited environments without GPUs.

The example dataset can be found in the `data/example_mri/` directory in this repository.

<br>

### Real Gene Expression Dataset for Answering Biological Questions Using the Tool:

I used AMP-AD MSBB RNA microarray data, obtained from the AD Knowledge Portal.

**Size and structure:** The working dataset consists of 20 publicly available RNA microarray samples, each containing ~40,000 gene-level probes. After preprocessing (duplicate removal, missing value imputation, scaling, PCA), the dimensionality is reduced while preserving 95% of the total variance. 

**Basic stats:**
- 20 samples
- Principal components (post-PCA) via PCA based on a 95% variance retention threshold
- Each sample originally contains ~40,000 gene expression values

**URL:** https://adknowledgeportal.synapse.org/

**Justification:**
This is a well-established, publicly available dataset used in Alzheimer's Disease (AD) research. It provides transcriptomic data directly relevant to studying disease-associated gene regulation and heterogeneity. The size is manageable for development and prototyping, while still retaining enough biological signal to validate generative models like adVAE.

**Biological question:**
Can adVAE generate synthetic gene expression profiles that retain biological characteristics of Alzheimer's gene expression patterns, even with very limited real data? Is this generated synthetic data useful for training classifier models that need to be trained on Alzheimer's-specific data?

**Expected results:**
We expect that the generated synthetic samples will:
- Have statistical properties (data distribution, Pearson's correlation, reconstruction accuracy) similar to real AD expression data
- Pass validation tests like T-tests and KS-tests
- Be potentially usable in downstream classification or enrichment tasks as augmentation data

<br>

### Real MRI Dataset for Answering Biological Questions Using the Tool:

I used WashU's OASIS-1 dataset containing structural MRI scans from mild cognitive impaired and Alzheimer's diagnosed individuals.

**Size and structure:** The selected dataset includes ~100 Gain-field corrected, atlas-registered average brain scans images, each in .img format. These are converted into 20 2D slices and processed (resized, converted to grayscale, normalised) into tensors via the adVAE MRI preprocessing pipeline.

**Basic stats:**
- ~100 subjects 
- 2D slices per subject used as input to the MRI-VAE model
- Converted into 128x128 grayscale normalised tensors

**URL:** https://sites.wustl.edu/oasisbrains/home/oasis-1/

**Justification:**
This dataset is a valuable real-world source of brain imaging data and offers diversity across Alzheimer’s progression stages. It enables testing whether adVAE can reconstruct and generate synthetic MRI slices that preserve key neurodegenerative patterns found in AD.

**Biological question:**
Can adVAE model and generate realistic 2D MRI slices that retain anatomical features associated with Alzheimer's Disease? Is this generated synthetic data useful for training classifier models that need to be trained on Alzheimer's-specific data?

**Expected results:**
We expect that the:
- Generated slices will preserve intensity and structure patterns typical of AD-affected regions
- Reconstruction performance will be evaluated using reconstruction accuracy, MSE, PSNR, SSIM, and visual comparison and will be similar to that of real AD MRI scans
- Synthetic slices will have statistical similarity (data distribution, Pearson's correlation) in pixel distribution and texture to real samples
- Synthetic slices will pass validation tests like T-tests and KS-tests

<br>

### Note:

Due to limited data availability, the datasets were not split into the standard 70-20-10 training-validation-testing ratio as initially intended. As a result, model performance was evaluated directly on the training data, which may limit the generalizability of the results. Future iterations of this project will focus on incorporating proper validation and test splits to enable more rigorous performance assessment. This will be critical for building a more robust and generalizable VAE model capable of generating high-quality synthetic multimodal biomedical data. 

