"""
Training and evaluation utilities for DON prediction models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.config.config import NN_CONFIG, DEVICE

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that gives more importance to samples with low DON values.
    """
    def __init__(self, threshold=1.0, weight_factor=2.0):
        """
        Args:
            threshold (float): DON value threshold below which samples get higher weights
            weight_factor (float): Factor by which to increase weights for low DON samples
        """
        super().__init__()
        self.threshold = threshold
        self.weight_factor = weight_factor
        
    def forward(self, pred, target):
        # Calculate base MSE
        mse = (pred - target) ** 2
        
        # Create weights: higher weights for low DON values
        weights = torch.where(target < self.threshold, 
                            self.weight_factor * torch.ones_like(target),
                            torch.ones_like(target))
        
        return (weights * mse).mean()

def train_neural_network_weighted(model, train_loader, val_loader, threshold=1.0, weight_factor=2.0, num_epochs=None, patience=None):
    """
    Train the neural network model with weighted loss function.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        threshold (float): DON value threshold below which samples get higher weights
        weight_factor (float): Factor by which to increase weights for low DON samples
        num_epochs (int, optional): Number of epochs to train
        patience (int, optional): Early stopping patience
        
    Returns:
        tuple: Trained model and training history
    """
    num_epochs = num_epochs or NN_CONFIG['num_epochs']
    patience = patience or NN_CONFIG['patience']
    
    criterion = WeightedMSELoss(threshold=threshold, weight_factor=weight_factor)
    optimizer = optim.Adam(model.parameters(), lr=NN_CONFIG['learning_rate'])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def train_neural_network(model, train_loader, val_loader, num_epochs=None, patience=None):
    """
    Train the neural network model.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int, optional): Number of epochs to train
        patience (int, optional): Early stopping patience
        
    Returns:
        tuple: Trained model and training history
    """
    num_epochs = num_epochs or NN_CONFIG['num_epochs']
    patience = patience or NN_CONFIG['patience']
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=NN_CONFIG['learning_rate'])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
            
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_model(model, X_train, X_test, y_train, y_test, is_neural_net=False):
    """
    Evaluate model performance on training and test sets.
    
    Args:
        model: Trained model (sklearn model or neural network)
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training targets
        y_test (np.ndarray): Test targets
        is_neural_net (bool): Whether the model is a neural network
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if is_neural_net:
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
            X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
            y_pred_train = model(X_train_tensor).cpu().numpy().ravel()
            y_pred_test = model(X_test_tensor).cpu().numpy().ravel()
    else:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    metrics = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test)
    }
    
    return metrics

def predict(model, X, is_neural_net=False):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model (sklearn model or neural network)
        X (np.ndarray): Input features
        is_neural_net (bool): Whether the model is a neural network
        
    Returns:
        np.ndarray: Predicted values
    """
    if is_neural_net:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            predictions = model(X_tensor).cpu().numpy().ravel()
    else:
        predictions = model.predict(X)
    
    return predictions

def objective(trial, model_class, train_loader, val_loader, is_weighted=False, weight_threshold=1.0):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        model_class: Neural network model class
        train_loader: Training data loader
        val_loader: Validation data loader
        is_weighted: Whether to use weighted loss
        weight_threshold: Threshold for weighted loss
        
    Returns:
        float: Validation loss
    """
    # Hyperparameters to optimize
    hidden_layers = [
        trial.suggest_int(f"hidden_layer_{i}", 32, 512, step=32)
        for i in range(3)  # 3 hidden layers
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_factor = trial.suggest_float("weight_factor", 1.5, 5.0) if is_weighted else 2.0
    
    # Create model with trial parameters
    model = model_class(input_dim=train_loader.dataset[0][0].shape[0])
    model.layer1 = nn.Linear(model.layer1.in_features, hidden_layers[0])
    model.layer2 = nn.Linear(hidden_layers[0], hidden_layers[1])
    model.layer3 = nn.Linear(hidden_layers[1], hidden_layers[2])
    model.layer4 = nn.Linear(hidden_layers[2], 1)
    model.dropout = nn.Dropout(dropout_rate)
    model = model.to(DEVICE)
    
    # Training setup
    if is_weighted:
        criterion = WeightedMSELoss(threshold=weight_threshold, weight_factor=weight_factor)
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = NN_CONFIG['patience']
    patience_counter = 0
    
    for epoch in range(NN_CONFIG['num_epochs']):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
        
        # Report intermediate value for Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def optimize_hyperparameters(model_class, train_loader, val_loader, is_weighted=False, 
                           weight_threshold=1.0, n_trials=100):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        model_class: Neural network model class
        train_loader: Training data loader
        val_loader: Validation data loader
        is_weighted: Whether to use weighted loss
        weight_threshold: Threshold for weighted loss
        n_trials: Number of optimization trials
        
    Returns:
        dict: Best hyperparameters
    """
    study = optuna.create_study(direction="minimize",
                               pruner=optuna.pruners.MedianPruner())
    
    study.optimize(lambda trial: objective(trial, model_class, train_loader, val_loader, 
                                         is_weighted, weight_threshold),
                  n_trials=n_trials)
    
    return study.best_params

def train_neural_network_with_optimal_params(model_class, train_loader, val_loader, params, 
                                           is_weighted=False, weight_threshold=1.0):
    """
    Train neural network with optimal hyperparameters.
    
    Args:
        model_class: Neural network model class
        train_loader: Training data loader
        val_loader: Validation data loader
        params: Dictionary of optimal hyperparameters
        is_weighted: Whether to use weighted loss
        weight_threshold: Threshold for weighted loss
        
    Returns:
        tuple: Trained model and training history
    """
    # Create model with optimal parameters
    model = model_class(input_dim=train_loader.dataset[0][0].shape[0])
    model.layer1 = nn.Linear(model.layer1.in_features, params['hidden_layer_0'])
    model.layer2 = nn.Linear(params['hidden_layer_0'], params['hidden_layer_1'])
    model.layer3 = nn.Linear(params['hidden_layer_1'], params['hidden_layer_2'])
    model.layer4 = nn.Linear(params['hidden_layer_2'], 1)
    model.dropout = nn.Dropout(params['dropout_rate'])
    model = model.to(DEVICE)
    
    # Training with optimal parameters
    if is_weighted:
        criterion = WeightedMSELoss(threshold=weight_threshold, 
                                  weight_factor=params.get('weight_factor', 2.0))
        return train_neural_network_weighted(model, train_loader, val_loader,
                                          threshold=weight_threshold,
                                          weight_factor=params.get('weight_factor', 2.0))
    else:
        return train_neural_network(model, train_loader, val_loader) 