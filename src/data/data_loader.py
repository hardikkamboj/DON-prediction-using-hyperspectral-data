"""
Data loading and preprocessing utilities for DON prediction models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.config.config import DATA_PATH, RANDOM_SEED, TEST_SIZE, BATCH_SIZE, DEVICE

class SpectralDataset(Dataset):
    """Custom PyTorch Dataset for spectral data."""
    
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.reshape(-1, 1))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    """
    Load and preprocess the data from CSV files.
    
    Returns:
        tuple: X and y numpy arrays
    """
    X = pd.read_csv(DATA_PATH['X_PROCESSED'])
    y = pd.read_csv(DATA_PATH['Y_PROCESSED'])
    
    return np.array(X), np.array(y)

def create_data_splits(X, y):
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        
    Returns:
        tuple: Training and testing data splits
    """
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

def create_data_loaders(X_train, X_test, y_train, y_test):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training targets
        y_test (np.ndarray): Testing targets
        
    Returns:
        tuple: Training and testing DataLoaders
    """
    train_dataset = SpectralDataset(X_train, y_train)
    test_dataset = SpectralDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader 