# Dataset Information:

This model will be trained on multimodal data.

### Gene Expression Data:

**Input:** Gene expression profiles (RNA microarray data) from AD individuals.

**Data Format:**
- .tsv file > tensors

**Data preprocessing:**
- Ensure all data is RNA microarray data obtained from human brain tissue of AD patients
- Handle missing values (NAs)
- Ensure samples are assigned a sample_id to prevent sample duplicates
- Ideal data split: 
    - 500 samples (example data - used for tool development)
    - remaining samples (70-20-10 into training, validation & test data)

**Sources:**
AD Knowledge Portal: https://adknowledgeportal.synapse.org/Explore/Programs/DetailsPage?Program=Resilience-AD

<br>

### MRI Data:

**Input:** Magnetic Resonance Imaging data obtained from AD individuals.

**Data Format:**
- Sagittal, Coronal, and Transverse slices in .gif format > tensors with each slice as a separate channe

**Data preprocessing:**
- Ensure all data is MRI images of the brains obtained from humans who have AD
- Ensure MRI images are assigned an image_id to prevent duplicates
- Ideal data split: 
    - 100-400 images (example data - used for tool development)
    - remaining samples (70-20-10 into training, validation & test data)

**Sources:**

WashU OASIS: https://sites.wustl.edu/oasisbrains/home/oasis-3/

<br>

### Example Data:

**Gene Expression data:**
I use a subset of gene expression data consisting of 5 samples, each represented by a .tsv format. The dataset will be preprocessed to handle missing values, normalize gene expression levels, and assign unique sample_ids to avoid duplicates as per the implemented pipeline. Basic stats: ~40,000 genes, 5 samples.

This example dataset is ideal for running adVAE because:
- It is small enough (< 20MB) to be used during development and testing.
- It captures the key data structure used in real AD studies (transcriptomics from human brain tissue).
- The dimensionality and complexity are sufficient to demonstrate the utility of a generative model like a VAE without requiring GPU acceleration.

The example dataset can be found in the data/example_gene_expression/ directory in this repository.

**MRI Data:**
I use a subset of MRI data consisting of 5 brain scan images, each in .gif format representing sagittal, coronal, and transverse slices. These images are preprocessed through the adVAE pipeline to ensure standardized orientation, resolution normalization, and unique image_ids for traceability. The resulting tensors are structured with each slice represented as a separate channel, maintaining spatial information across modalities.

This example dataset is ideal for running adVAE because:
- It is lightweight and < 20MB, making it suitable for rapid development and debugging.
- It reflects the typical structure of 3D neuroimaging data used in Alzheimer's research.
- It enables demonstration of image-based generative modeling using VAEs, even on resource-limited environments without GPUs.

The example dataset can be found in the data/example_mri/ directory in this repository.