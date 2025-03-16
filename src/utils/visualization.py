"""
Visualization utilities for plotting model results and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_training_history(history):
    """
    Plot training and validation loss history.
    
    Args:
        history (dict): Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predictions(y_train, y_train_pred, y_test, y_test_pred, title):
    """
    Plot predicted vs actual values for both training and test sets.
    
    Args:
        y_train (np.ndarray): True training values
        y_train_pred (np.ndarray): Predicted training values
        y_test (np.ndarray): True test values
        y_test_pred (np.ndarray): Predicted test values
        title (str): Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Training set plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Actual log(DON)')
    plt.ylabel('Predicted log(DON)')
    plt.title(f'{title} - Training Set')
    
    # Test set plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual log(DON)')
    plt.ylabel('Predicted log(DON)')
    plt.title(f'{title} - Test Set')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(models_results):
    """
    Plot comparison of different models' performance.
    
    Args:
        models_results (list): List of tuples containing (model_name, metrics)
    """
    comparison_df = pd.DataFrame({
        'Model': [name for name, _ in models_results],
        'Train RMSE': [results['rmse_train'] for _, results in models_results],
        'Test RMSE': [results['rmse_test'] for _, results in models_results],
        'Train R²': [results['r2_train'] for _, results in models_results],
        'Test R²': [results['r2_test'] for _, results in models_results],
        'Train MAE': [results['mae_train'] for _, results in models_results],
        'Test MAE': [results['mae_test'] for _, results in models_results]
    })
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='Test RMSE', data=comparison_df)
    plt.title('Test RMSE by Model')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='Model', y='Test R²', data=comparison_df)
    plt.title('Test R² by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_indices, coefficients, top_n=10):
    """
    Plot feature importance from Lasso regression or XGBoost.
    
    Args:
        feature_indices (np.ndarray): Array of feature indices
        coefficients (np.ndarray): Array of feature coefficients/importance
        top_n (int): Number of top features to plot
    """
    feature_importance_df = pd.DataFrame({
        'Feature Index': feature_indices,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value to get top features
    feature_importance_df['Abs Coefficient'] = np.abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.nlargest(top_n, 'Abs Coefficient')
    
    # Sort by actual coefficient value for visualization
    feature_importance_df = feature_importance_df.sort_values('Coefficient')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(y='Abs Coefficient', x='Feature Index', 
                data=feature_importance_df,
                palette='coolwarm')
    
    plt.title(f'Top {top_n} Important Wavelengths')
    plt.ylabel('Absolute Coefficient value')
    plt.xlabel('Feature Index')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show() 