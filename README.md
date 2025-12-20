# Diabetes Prediction Project

## Project Overview

This project implements a comprehensive machine learning pipeline to predict diabetes in women using the Pima Indians Diabetes dataset. The implementation includes multiple classification algorithms with complete evaluation and comparison.

## Dataset Description

**Pima Indians Diabetes Dataset**
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
   - Train 7 classification algorithms
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
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Gradient Boosting (XGBoost)
7. Naive Bayes

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
- `models/best_model.pkl` - Best performing model

## Key Findings

*(Results will be populated after running the notebooks)*

## References

- Dataset Source: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Course: CMPE403 Data Science & Analytics
