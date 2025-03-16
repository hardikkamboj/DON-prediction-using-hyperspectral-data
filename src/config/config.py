"""
Configuration settings for the DON prediction models.
"""

import torch
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Data settings
DATA_PATH = {
    'X_PROCESSED': 'x_processed.csv',
    'Y_PROCESSED': 'y_processed.csv'
}

# Training settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32

# Neural Network settings
NN_CONFIG = {
    'hidden_layers': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'num_epochs': 200,
    'patience': 30
}

# Model paths
MODEL_PATHS = {
    'LINEAR': 'models/linear_model.pkl',
    'RIDGE': 'models/ridge_model.pkl',
    'LASSO': 'models/lasso_model.pkl',
    'XGBOOST': 'models/xgboost_model.pkl',
    'NEURAL_NET': 'models/neural_network.pth',
    'SCALER': 'models/scaler.pkl'  # To save the data scaler
}

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cross validation settings
CV_CONFIG = {
    'n_splits': 5,
    'cv_epochs': 50,
    'cv_patience': 10
}

# Regularization parameters
REGULARIZATION = {
    'ridge_alphas': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    'lasso_alphas': [1e-4, 1e-3, 1e-2, 1e-1, 1]
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'verbose': 1
}