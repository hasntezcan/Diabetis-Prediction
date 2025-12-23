"""
Configuration file for Diabetes Prediction Project
Contains all configuration parameters and constants
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# Data file paths
DATA_PATH = os.path.join(DATA_DIR, 'diabetes.csv')

# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split ratio
TEST_SIZE = 0.2

# Feature names
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# Target variable name
TARGET_NAME = 'Outcome'

# Features with zeros that should be treated as missing values
ZERO_AS_MISSING_FEATURES = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# Cross-validation parameters
CV_FOLDS = 5

# Model parameters (default hyperparameters)
MODEL_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'decision_tree': {
        'random_state': RANDOM_STATE,
        'max_depth': 10
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'max_depth': 10
    },
    'knn': {
        'n_neighbors': 5
    },
    'naive_bayes': {}
}

# Hyperparameter tuning grids
# Updated with stronger regularization to prevent overfitting
PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['saga', 'liblinear']
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10],  # Reduced maximum depth
        'min_samples_split': [5, 10, 20],  # Increased minimum
        'min_samples_leaf': [2, 4, 8],  # Increased minimum
        'max_features': ['sqrt', 'log2', None]  # Added feature subsampling
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],  # More conservative depths
        'min_samples_split': [10, 20, 30],  # Stronger constraints
        'min_samples_leaf': [4, 8, 12],  # Higher minimums
        'max_features': ['sqrt', 'log2']  # Feature subsampling for diversity
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Visualization settings
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 100
SEABORN_STYLE = 'whitegrid'
COLOR_PALETTE = 'Set2'
