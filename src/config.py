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
    'svm': {
        'kernel': 'rbf',
        'random_state': RANDOM_STATE,
        'probability': True
    },
    'knn': {
        'n_neighbors': 5
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'naive_bayes': {}
}

# Hyperparameter tuning grids
PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
}

# Visualization settings
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 100
SEABORN_STYLE = 'whitegrid'
COLOR_PALETTE = 'Set2'
