"""
Model architectures for DON prediction.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from src.config.config import NN_CONFIG, DEVICE, REGULARIZATION, XGBOOST_PARAMS

class NeuralNetwork(nn.Module):
    """Neural Network model for DON prediction."""
    
    def __init__(self, input_dim):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Number of input features
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, NN_CONFIG['hidden_layers'][0])
        self.layer2 = nn.Linear(NN_CONFIG['hidden_layers'][0], NN_CONFIG['hidden_layers'][1])
        self.layer3 = nn.Linear(NN_CONFIG['hidden_layers'][1], NN_CONFIG['hidden_layers'][2])
        self.layer4 = nn.Linear(NN_CONFIG['hidden_layers'][2], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(NN_CONFIG['dropout_rate'])
        
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = self.layer4(x)
        return x

class LinearModels:
    """Class containing linear regression models with different regularizations."""
    
    @staticmethod
    def get_linear_regression():
        """Get simple linear regression model."""
        return LinearRegression()
    
    @staticmethod
    def get_ridge_regression(cv=5):
        """
        Get Ridge regression model with cross-validation.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            GridSearchCV: Ridge regression model with best parameters
        """
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)
        param_grid = {'alpha': REGULARIZATION['ridge_alphas']}
        return GridSearchCV(
            Ridge(random_state=42),
            param_grid,
            cv=cv_split,
            scoring='neg_mean_squared_error'
        )
    
    @staticmethod
    def get_lasso_regression(cv=5):
        """
        Get Lasso regression model with cross-validation.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            GridSearchCV: Lasso regression model with best parameters
        """
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)
        param_grid = {'alpha': REGULARIZATION['lasso_alphas']}
        return GridSearchCV(
            Lasso(random_state=42, max_iter=10000, tol=1e-2),
            param_grid,
            cv=cv_split,
            scoring='neg_mean_squared_error'
        )

class XGBoostModel:
    """XGBoost model with hyperparameter tuning."""
    
    @staticmethod
    def get_model(cv=5):
        """
        Get XGBoost model with cross-validation.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            GridSearchCV: XGBoost model with best parameters
        """
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Create parameter grid
        param_grid = {
            'max_depth': XGBOOST_PARAMS['max_depth'],
            'learning_rate': XGBOOST_PARAMS['learning_rate'],
            'n_estimators': XGBOOST_PARAMS['n_estimators'],
            'min_child_weight': XGBOOST_PARAMS['min_child_weight'],
            'subsample': XGBOOST_PARAMS['subsample'],
            'colsample_bytree': XGBOOST_PARAMS['colsample_bytree'],
            'gamma': XGBOOST_PARAMS['gamma']
        }
        
        # Create base model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        return GridSearchCV(
            xgb_model,
            param_grid,     
            cv=cv_split,
            scoring='neg_mean_squared_error',
            verbose=XGBOOST_PARAMS['verbose'] # so that you don't get bored while model is being trained
        ) 