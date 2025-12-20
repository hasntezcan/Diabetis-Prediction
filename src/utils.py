"""
Utility functions for Diabetes Prediction Project
Contains reusable functions for data loading, preprocessing, modeling, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import joblib
import os
from typing import Tuple, Dict, Any
import config


def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load the diabetes dataset
    
    Args:
        file_path: Path to the CSV file (default: config.DATA_PATH)
    
    Returns:
        pandas DataFrame with the dataset
    """
    if file_path is None:
        file_path = config.DATA_PATH
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle zeros as missing values in medical measurements
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('median' or 'mean')
    
    Returns:
        DataFrame with imputed values
    """
    df_copy = df.copy()
    
    for feature in config.ZERO_AS_MISSING_FEATURES:
        # Replace zeros with NaN
        df_copy[feature] = df_copy[feature].replace(0, np.nan)
        
        # Impute missing values
        if strategy == 'median':
            df_copy[feature].fillna(df_copy[feature].median(), inplace=True)
        elif strategy == 'mean':
            df_copy[feature].fillna(df_copy[feature].mean(), inplace=True)
    
    return df_copy


def preprocess_data(df: pd.DataFrame, test_size: float = None, random_state: int = None) -> Tuple:
    """
    Preprocess data: handle missing values, scale features, and split
    
    Args:
        df: Input DataFrame
        test_size: Test set size (default: config.TEST_SIZE)
        random_state: Random seed (default: config.RANDOM_STATE)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    # Handle missing values
    df_processed = handle_missing_values(df)
    
    # Separate features and target
    X = df_processed[config.FEATURE_NAMES]
    y = df_processed[config.TARGET_NAME]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, X_train, y_train):
    """
    Train a classification model
    
    Args:
        model: Sklearn classifier object
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Fitted model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """
    Evaluate model performance with multiple metrics
    
    Args:
        model: Trained sklearn classifier
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    
    Returns:
        Dictionary with performance metrics
    """
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probabilities for ROC
    if hasattr(model, 'predict_proba'):
        y_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = model.decision_function(X_test)
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1_score': f1_score(y_test, y_pred_test),
        'roc_auc': roc_auc_score(y_test, y_proba_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CV_FOLDS, scoring='accuracy')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    return metrics


def plot_confusion_matrix(cm, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix as heatmap
    
    Args:
        cm: Confusion matrix array
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_test, y_proba, model_name='Model', save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        model_name: Name for the plot
        save_path: Path to save figure (optional)
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.show()


def compare_models(results_df: pd.DataFrame, metric='test_accuracy', save_path=None):
    """
    Create bar plot comparing models
    
    Args:
        results_df: DataFrame with model results
        metric: Metric to compare
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 6))
    results_sorted = results_df.sort_values(by=metric, ascending=False)
    
    sns.barplot(data=results_sorted, x='model', y=metric, palette=config.COLOR_PALETTE)
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xlabel('Model')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.show()


def save_model(model, filename: str, directory: str = None):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        filename: Filename for the model
        directory: Directory to save (default: config.MODELS_DIR)
    """
    if directory is None:
        directory = config.MODELS_DIR
    
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filename: str, directory: str = None):
    """
    Load trained model from disk
    
    Args:
        filename: Filename of the model
        directory: Directory where model is saved (default: config.MODELS_DIR)
    
    Returns:
        Loaded model
    """
    if directory is None:
        directory = config.MODELS_DIR
    
    filepath = os.path.join(directory, filename)
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def print_classification_metrics(model_name: str, metrics: Dict[str, Any]):
    """
    Print formatted classification metrics
    
    Args:
        model_name: Name of the model
        metrics: Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"Train Accuracy:     {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:      {metrics['test_accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-Score:           {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"CV Mean (±std):     {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
    print(f"{'='*60}\n")
