"""
Simple utility functions for loading and using saved models.
"""

import torch
from src.models.models import NeuralNetwork
from src.config.config import DEVICE

def load_model_for_prediction(model_path, input_dim):
    """
    Load a saved model for prediction.
    
    Args:
        model_path (str): Path to the saved model file
        input_dim (int): Input dimension of the model
        
    Returns:
        model: Loaded neural network model
    """
    # Initialize model with known architecture
    model = NeuralNetwork(input_dim).to(DEVICE)
    
    # Load the saved model info
    model_info = torch.load(model_path)
    
    # Load the state dict
    model.load_state_dict(model_info['state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def predict_with_model(model, X):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded neural network model
        X (numpy.ndarray): Input features
        
    Returns:
        numpy.ndarray: Predicted values
    """
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        predictions = model(X_tensor).cpu().numpy().ravel()
    return predictions

# Example usage
if __name__ == "__main__":
    import numpy as np
    from src.config.config import MODEL_PATHS
    
    # Example data
    X = np.random.rand(5, 448)  # 5 samples, 448 features
    
    # Load and use model
    model = load_model_for_prediction(MODEL_PATHS['NEURAL_NET'], input_dim=448)
    predictions = predict_with_model(model, X)
    print("Predictions:", predictions) 