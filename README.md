# Machine Learning Pharmaceutical Therapy Recommender System for Schizophrenia Spectrum Disorders

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the implementation of a machine learning-based medication recommender system for hospital in-patients with schizophrenia spectrum disorders. The system uses collaborative filtering and distance-based representation methods to recommend antipsychotic medications based on patient demographics, clinical characteristics, and treatment history.

**Paper:** "Developing and Validating a Machine Learning Pharmaceutical Therapy Recommender System for Hospital In-Patients with Schizophrenia Spectrum Disorders"

**Authors:** Maximin Lange, Urvik Metha, Nikolaos Koutsouleris, Feras Fayez, Ricardo Twumasi



## Installation

### Prerequisites

- Python 3.8 or higher
- Access to MIMIC-IV, MIMIC-III, and Northwestern ICU databases via PhysioNet
- Git

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ml-pharmacy-recommender.git
cd ml-pharmacy-recommender
```

2. **Install dependencies:**
```bash
# Core ML libraries
pip install metric-learn surprise gower skrebate tabulate

# Standard data science stack  
pip install pandas numpy scikit-learn scipy matplotlib seaborn



3. **Set up database access:**
- Complete PhysioNet CITI training for all three datasets
- Download and set up local database connections
- Update database paths in the extraction scripts as needed

## Data Access

This project uses three medical databases:

### MIMIC-IV (Primary Development Dataset)
- **Access**: Requires PhysioNet credentialed access
- **URL**: https://physionet.org/content/mimiciv/
- **Requirements**: Complete CITI training and sign data use agreement
- **Time Period**: 2008-2022 from Beth Israel Deaconess Medical Center

### MIMIC-III (Temporal Validation)
- **Access**: Requires PhysioNet credentialed access  
- **URL**: https://physionet.org/content/mimiciii/
- **Requirements**: Complete CITI training and sign data use agreement
- **Time Period**: 2001-2012 (CareVue subset only) from Beth Israel Deaconess Medical Center

### Northwestern ICU (Geographic Validation)
- **Access**: Requires PhysioNet credentialed access
- **URL**: https://physionet.org/content/nwicu/
- **Requirements**: Complete CITI training and sign data use agreement
- **Time Period**: 2020-2021 from Northwestern Memorial HealthCare

## Project Structure

```
ml-pharmacy-recommender/
│
├── LICENSE                                    # MIT License
├── README.md                                  # This file
│
├── data_extraction.sql                        # MIMIC-IV data extraction queries
├── data_extraction_MM_NWICU.py               # Northwestern ICU data extraction
├── data_extraction_validation_MM3            # MIMIC-III data extraction
│
├── preprocessing.py                           # Data cleaning and preprocessing
├── hyperparameter_tuning_finding_best_model.py # Model selection and optimization
│
├── validate_MM_NWICU.py                      # Northwestern external validation
├── validate_mimic3.py                        # MIMIC-III temporal validation
│
├── requirements.txt                          # Python dependencies
└── saved_models/                             # Trained model files (not included)
    ├── cf_cosine_k7_model.pkl
    └── cf_cosine_k7_train_data.pkl
```

## Usage

### Complete Workflow

Follow these steps to reproduce the study results:

#### 1. **Data Extraction**
```bash
# Extract MIMIC-IV training data (requires database access)
python data_extraction.sql  # Run in your SQL environment

# Extract Northwestern validation data
python data_extraction_MM_NWICU.py

# Extract MIMIC-III temporal validation data  
python data_extraction_validation_MM3
```

#### 2. **Data Preprocessing**
```bash
# Clean and preprocess all datasets
python preprocessing.py
```
This script handles:
- ICD-9 to ICD-10 code mapping
- Race/ethnicity standardization  
- Antipsychotic medication name standardization
- Feature engineering (affinity scores, comorbidity counts)

#### 3. **Model Training & Hyperparameter Tuning**
```bash
# Find optimal model and hyperparameters
python hyperparameter_tuning_finding_best_model.py
```
This will:
- Test 5 different algorithms (CF-Cosine, CF-Euclidean, DR-RBA, DR-RBF, DR-NCA)
- Use nested 5-fold cross-validation
- Save the best model (CF-Cosine with K=7) to `saved_models/`

#### 4. **External Validation**
```bash
# Geographic validation on Northwestern data
python validate_MM_NWICU.py

# Temporal validation on MIMIC-III data  
python validate_mimic3.py
```

### Expected Results

After running the complete pipeline, you should see:

**Internal Validation (MIMIC-IV):**
- Overall RMSE: 0.116, MAP@3: 0.490
- Visit 1: RMSE: 0.117, MAP@3: 0.702

**External Geographic Validation (Northwestern):**
- Overall RMSE: 0.497, MAP@3: 0.432
- Visit 1: RMSE: 0.472, MAP@3: 0.375

**External Temporal Validation (MIMIC-III):**
- Overall RMSE: 0.578, MAP@3: 0.359
- Visit 2: RMSE: 0.519, MAP@3: 0.813

## Methodology

### Affinity Score Calculation

The treatment outcome is measured using a composite affinity score (0-1 range):

```python
def calculate_affinity_score(patient_data):
    """
    Affinity = (T + V + S + L) / 4 - P
    
    Where:
    T: normalized time between admissions
    V: normalized number of visits  
    S: normalized time between medication switches
    L: normalized total length of stay
    P: penalty for switching medications
    """
    # Implementation details in src/utils/affinity_score.py
```

### Algorithms Implemented

1. **Collaborative Filtering**:
   - Cosine Similarity (CF-Cosine)
   - Euclidean Distance (CF-Euclidean)

2. **Distance-Based Representation**:
   - Gower Similarity with Relief-Based Attribute weighting (DR-RBA)
   - Gaussian RBF transformations (DR-RBF)  
   - Neighborhood Component Analysis (DR-NCA)

### Evaluation Metrics

- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAP@3**: Mean Average Precision at position 3
- **Coverage**: Proportion where actual therapy not in top 3
- **Overlap**: Proportion where actual therapy in top 3

## Reproducing Results

### Key Scripts

| Script | Purpose | Expected Runtime |
|--------|---------|------------------|
| `preprocessing.py` | Data cleaning and feature engineering | ~5 minutes |
| `hyperparameter_tuning_finding_best_model.py` | Model selection via nested CV | ~10 hours |
| `validate_MM_NWICU.py` | Geographic external validation | ~5 minutes |
| `validate_mimic3.py` | Temporal external validation | ~5 minutes |



### Dependencies

Install required packages:
```bash
pip install metric-learn surprise gower skrebate tabulate
pip install pandas numpy scikit-learn scipy
```






## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Corresponding Author**: Maximin Lange (maximin.lange@kcl.ac.uk)
- **Institution**: Institute of Psychiatry, Psychology & Neuroscience, King's College London


**Data Use Agreements**: Required for accessing MIMIC and NWICU datasets through PhysioNet.

- Enhanced interpretability and explainability features
