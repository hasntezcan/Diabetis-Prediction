# Diabetes Prediction Project - Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
cd "/Users/hasantezcan/Diabet Prediction Project/Diabetis-Prediction"
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

#### Step 1: Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```
**What it does**: Analyzes the dataset, visualizes distributions, identifies missing values, creates correlation matrices

#### Step 2: Data Preprocessing
```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```
**What it does**: Handles missing values, scales features, splits data into train/test sets

#### Step 3: Model Training & Evaluation
```bash
jupyter notebook notebooks/03_model_training_evaluation.ipynb
```
**What it does**: Trains 5 classification models, evaluates performance, compares results

#### Step 4: Hyperparameter Tuning
```bash
jupyter notebook notebooks/04_hyperparameter_tuning.ipynb
```
**What it does**: Optimizes top models using GridSearch, selects final best model

## 📊 What You'll Get

### Visualizations (16+ figures saved in `results/figures/`)
- Missing value analysis
- Target distribution charts
- Feature distributions and box plots
- Correlation heatmap
- Pair plots
- Model comparison charts
- ROC curves
- Confusion matrices
- Feature importance plots

### Models Trained
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. KNN
5. Naive Bayes

### Saved Models
- `models/best_model.pkl` - Best model from initial training
- `models/best_model_tuned.pkl` - Optimized final model
- `models/scaler.pkl` - Feature scaler for predictions

### Results
- `results/model_performance.csv` - All model metrics
- `results/best_model_params.csv` - Optimized hyperparameters

## 💡 Key Features

✅ **Complete ML Pipeline**: From raw data to deployed model
✅ **5 Algorithms**: Comprehensive model comparison
✅ **Hyperparameter Tuning**: GridSearchCV optimization
✅ **Rich Visualizations**: 16+ professional charts
✅ **Reusable Code**: Utility functions in `src/utils.py`
✅ **Well Documented**: README, docstrings, and comments

## 🎯 Expected Performance

Based on the Pima Indians Diabetes dataset:
- **Accuracy**: 70-80%
- **ROC-AUC**: 0.75-0.85
- **Best Models**: Usually Random Forest or Logistic Regression

## 📝 Project Structure
```
Diabetis-Prediction/
├── notebooks/          # 4 Jupyter notebooks (sequential workflow)
├── src/                # Python modules (config.py, utils.py)
├── data/               # Dataset (diabetes.csv)
├── models/             # Saved models
├── results/            # Metrics and visualizations
└── requirements.txt    # Dependencies
```

## 🔍 Verification

Check everything is working:
```bash
# Verify Python files exist
ls -l src/*.py

# Verify notebooks exist
ls -l notebooks/*.ipynb

# Check if dataset is present
ls -l data/diabetes.csv
```

## 📚 Documentation

- **README.md**: Complete project overview
- **Walkthrough.md**: Detailed implementation guide (in artifacts)
- **Implementation Plan**: Original design document (in artifacts)

---

**Ready to start!** Open the first notebook and run cells sequentially. All code is fully functional and ready to execute.
