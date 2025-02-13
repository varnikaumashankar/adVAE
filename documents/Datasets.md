# Dataset Information:

This model will be trained on multimodal data.

### Gene Expression Data:

**Input:** Gene expression profiles (transcriptomics data) from AD individuals.

**Data Format:**
- .csv file with genes and samples as rows and columns > 2D tensors

**Data preprocessing:**
- Ensure all data is RNA microarray data obtained from human brain tissue of AD patients
- Handle missing values (NAs)
- Ensure samples are assigned a sample_id to prevent sample duplicates
- Ideal data split: 
    - 500 samples (example data - used for tool development)
    - remaining samples (70-20-10 into training, validation & test data)

**Sources:**

NCBI GEO: https://www.ncbi.nlm.nih.gov/gds/
EMBL-EBI: https://www.ebi.ac.uk/biostudies/arrayexpress/studies
AD Knowledge Portal: https://adknowledgeportal.synapse.org/Explore/Programs/DetailsPage?Program=Resilience-AD

<br>

### EEG Data:

**Input:** Electroencephalogram data obtained from AD individuals.

**Data Format:**
- .edf file with successive rows corresponding to successive time slices and each column of the data file corresponding to to an individual sensor location or other information tag.

**Data preprocessing:**
- Ensure all data is EEG time-series data obtained from humans who are AD patients
Handle missing values (NAs)
- Ensure EEG samples are assigned a sample_id to prevent sample duplicates
- Ideal data split: 
    - 300-500 samples (example data - used for tool development)
    - remaining samples (70-20-10 into training, validation & test data)

**Sources:**

OpenNeuro: https://openneuro.org/datasets/ds004504/versions/1.0.8
TUH EEG corpus: https://isip.piconepress.com/projects/tuh_eeg/

<br>

### MRI Data:

**Input:** Magnetic Resonance Imaging data obtained from AD individuals.

**Data Format:**
- NIfTI format or .nii format 

**Data preprocessing:**
- Ensure all data is MRI images of the brains obtained from humans who have AD
- Ensure MRI images are assigned an image_id to prevent duplicates
- Ideal data split: 
    - 100-400 images (example data - used for tool development)
    - remaining samples (70-20-10 into training, validation & test data)

**Sources:**

WashU OASIS: https://sites.wustl.edu/oasisbrains/home/oasis-3/
OpenNeuro: https://openneuro.org/datasets/ds004504/versions/1.0.8