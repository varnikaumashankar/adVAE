# Dataset Information:

This model will be trained on multimodal data.

### Gene Expression Data:

**Input:** Gene expression profiles (RNA microarray data) from AD individuals.

**Data Format:**
- .tsv file > tensors

**Data Preprocessing:**
- Ensure all data is RNA microarray data obtained from human brain tissue of AD patients
- Handle missing values (NAs)
- Ensure samples are assigned a sample_id to prevent sample duplicates
- Ideal data split: 
    - 500 samples (example data - used for tool development)
    - Remaining samples (70-20-10 into training, validation & test data)

**Sources:**
AD Knowledge Portal: https://adknowledgeportal.synapse.org/Explore/Programs/DetailsPage?Program=Resilience-AD

<br>

### MRI Data: 

**Input:** Magnetic Resonance Imaging data obtained from AD individuals.

**Data Format:**
- Transverse slices extracted from Gain-field corrected, atlas-registered average brain scans (.img) > tensors (128x128 resolution per slice)

**Data Preprocessing:**
- Ensure all data is MRI images of the brains obtained from humans who have AD
- Standardize resolution to 128x128 pixels
- Assign image_id to prevent duplicates and enable traceability
- Normalize intensities, ensure proper orientation, and extract 2D slices from 3D volumes
- Ideal data split: 
    - 100-400 images (example data - used for tool development)
    - Remaining samples (70-20-10 into training, validation & test data)

**Sources:**
WashU OASIS: https://sites.wustl.edu/oasisbrains/home/oasis-1/

<br>

### Example Data:

**Gene Expression Data:**
I use a subset of gene expression data consisting of 5 samples, each represented by a .tsv format. The dataset will be preprocessed to handle missing values, normalize gene expression levels, and assign unique sample_ids to avoid duplicates as per the implemented pipeline. Basic stats: ~40,000 genes, 5 samples.

This example dataset is ideal for running adVAE because:
- It is small enough (< 20MB) to be used during development and testing.
- It captures the key data structure used in real AD studies (transcriptomics from human brain tissue).
- The dimensionality and complexity are sufficient to demonstrate the utility of a generative model like a VAE without requiring GPU acceleration.

The example dataset can be found in the data/example_gene_expression/ directory in this repository.

**MRI Data:**
I use a subset of MRI data consisting of 5 Gain-field corrected, atlas-registered average brain scans images, each in .img format that was used to extract transverse slices. These images are preprocessed through the adVAE pipeline to ensure standardized orientation, resolution normalization, and unique image_ids for traceability. The resulting tensors are structured with each slice represented as a separate channel, maintaining spatial information across modalities.

This example dataset is ideal for running adVAE because:
- It is lightweight and < 20MB, making it suitable for rapid development and debugging.
- It reflects the typical structure of 3D neuroimaging data used in Alzheimer's research.
- It enables demonstration of image-based generative modeling using VAEs, even on resource-limited environments without GPUs.

The example dataset can be found in the data/example_mri/ directory in this repository.

<br>

### Real Gene Expression Dataset for Answering Biological Questions Using the Tool:

I used AMP-AD MSBB RNA microarray data, obtained from the AD Knowledge Portal.

**Size and structure:** The working dataset consists of 20 publicly available RNA microarray samples, each containing ~40,000 gene-level probes. After preprocessing (duplicate removal, misisng value imputation, scaling, PCA), the dimensionality is reduced while preserving 95% of the total variance. 

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
- Have statistical properties (distribution, correlation) similar to real AD expression data
- Pass validation tests like t-tests, correlation, and reconstruction accuracy
- Be potentially usable in downstream classification or enrichment tasks as augmentation data

<br>

### Real Gene Expression Dataset for Answering Biological Questions Using the Tool:

I used OASIS-1 structural MRI scans from mild cognitive impaired and Alzheimer's diagnosed individuals.

**Size and structure:** The selected dataset includes ~100 Gain-field corrected, atlas-registered average brain scans images, each in .img format. These are converted into 2D slices and normalized tensors via the adVAE MRI preprocessing pipeline.

**Basic stats:**
- ~100 subjects
- 2D slices per subject used as input to the MRI-VAE model
- Converted into 128x128 grayscale tensors

**URL:** https://sites.wustl.edu/oasisbrains/home/oasis-1/

**Justification:**
This dataset is a valuable real-world source of brain imaging data and offers diversity across Alzheimer’s progression stages. It enables testing whether adVAE can reconstruct and generate synthetic MRI slices that preserve key neurodegenerative patterns found in AD.

**Biological question:**
Can adVAE model and generate realistic 2D MRI slices that retain anatomical features associated with Alzheimer's Disease? Is this generated synthetic data useful for training classifier models that need to be trained on Alzheimer's-specific data?

**Expected results:**
- Generated slices will preserve intensity and structure patterns typical of AD-affected regions
- Reconstruction performance will be evaluated using MSE, SSIM, and visual comparison
- Synthetic slices will have statistical similarity in pixel distribution and texture to real samples

