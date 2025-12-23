# Diabetes Prediction Project

## Project Overview

This project implements a comprehensive machine learning pipeline to predict diabetes in women using the Pima Indians Diabetes dataset. The implementation includes multiple classification algorithms with complete evaluation and comparison.

## Dataset Description

**Diabetes Dataset**
- **Samples**: 768 women
- **Features**: 8 medical measurements
  - Pregnancies: Number of times pregnant
  - Glucose: Plasma glucose concentration (mg/dL)
  - BloodPressure: Diastolic blood pressure (mm Hg)
  - SkinThickness: Triceps skin fold thickness (mm)
  - Insulin: 2-Hour serum insulin (mu U/ml)
  - BMI: Body mass index (weight in kg/(height in m)^2)
  - DiabetesPedigreeFunction: Diabetes pedigree function
  - Age: Age (years)
- **Target**: Outcome (0 = No Diabetes, 1 = Diabetes)

## Project Structure

```
Diabetis-Prediction/
├── data/
│   └── diabetes.csv              # Original dataset
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_hyperparameter_tuning.ipynb
├── src/
│   ├── config.py                 # Configuration parameters
│   └── utils.py                  # Utility functions
├── models/
│   └── best_model.pkl            # Saved best model
├── results/
│   ├── model_performance.csv     # Performance metrics
│   └── figures/                  # Visualization outputs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in sequence:

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```
   - Understand data distribution
   - Identify missing values and outliers
   - Visualize feature relationships

2. **Data Preprocessing**
   ```bash
   jupyter notebook notebooks/02_data_preprocessing.ipynb
   ```
   - Handle missing values (zeros)
   - Apply feature scaling
   - Split into train/test sets

3. **Model Training & Evaluation**
   ```bash
   jupyter notebook notebooks/03_model_training_evaluation.ipynb
   ```
   - Train 5 classification algorithms
   - Evaluate with multiple metrics
   - Compare model performance

4. **Hyperparameter Tuning**
   ```bash
   jupyter notebook notebooks/04_hyperparameter_tuning.ipynb
   ```
   - Optimize top models
   - Save best model

## Models Implemented

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Naive Bayes

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Cross-Validation Scores

## Results

Model performance results are saved in:
- `results/model_performance.csv` - Detailed metrics for all models
- `results/figures/` - Visualization outputs


## References

- Dataset Source: [[Diabetes Database](https://www.kaggle.com/datasets/mathchi/diabetes-data-set?phase=FinishSSORegistration&returnUrl=/datasets/mathchi/diabetes-data-set/versions/1?resource=download&SSORegistrationToken=CfDJ8KYR5TFjzfZErZMgR3fFdNgvfNt38gEIw4uo4h2ad3wET68hN9VcMrbj4OQ5aeyf1UgZmwuyQiBK3VM0Lj0ZPe1eZdZ2GmGJdre1ttHejg6CYe8ual7BNu1llTBtibI4aZPGAo_Jw79C9-QKv5VSbIhEPzwr7_L0aoC1WvKPHtoNErHDSnzmOYEY-NCR9WbQPUNYNGS7EHROYR94F-16bwYZ-TnspHgpdYlfMlTOdFVpDSfwLw2uqD9qleiemgki6xbjHVjE3OXbxFIE5tEB6oXMZmGKHhNUc4TqkUcob1XtbtQj7cht_45L7i_j7EG5R_kLaDDKRfgN6ph1HYyCFBBM14dk4g&DisplayName=Hasan%20Tezcan)]
