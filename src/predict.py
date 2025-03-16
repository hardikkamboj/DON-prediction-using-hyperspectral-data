"""
Script for making predictions using trained models.
"""

import numpy as np
import pandas as pd
import torch
import joblib
import argparse
from src.models.models import NeuralNetwork
from src.config.config import MODEL_PATHS, DEVICE

def preprocess_data(X):
    """
    Preprocess the input data using the saved scaler.
    
    Args:
        X (pd.DataFrame): Input features
        
    Returns:
        np.ndarray: Preprocessed features
    """
    try:
        # Load scaler
        print("Loading scaler...")
        scaler = joblib.load(MODEL_PATHS['SCALER'])
        
        # Scale the input data
        print("Normalizing input data...")
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    except FileNotFoundError:
        raise FileNotFoundError(
            "Scaler not found! Make sure you have trained the models first "
            "using main.py which saves the scaler."
        )
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

def load_model(model_type):
    """
    Load a trained model.
    
    Args:
        model_type (str): Type of model to load ('linear', 'ridge', 'lasso', 'xgboost', or 'neural_net')
        
    Returns:
        object: Loaded model
    """
    model_type = model_type.upper()
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        if model_type == 'NEURAL_NET':
            # For Neural Network, we need to initialize the model first
            model = NeuralNetwork(input_dim=None)  # Will be set when data is loaded
            model.load_state_dict(torch.load(MODEL_PATHS[model_type]))
            model.to(DEVICE)
            model.eval()
        else:
            # For other models, we can load directly
            model = joblib.load(MODEL_PATHS[model_type])
        
        return model
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found! Make sure you have trained the {model_type} model first "
            "using main.py which saves all models."
        )

def predict(model, X, model_type):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded model
        X (np.ndarray): Input features
        model_type (str): Type of model ('linear', 'ridge', 'lasso', 'xgboost', or 'neural_net')
        
    Returns:
        np.ndarray: Predictions
    """
    model_type = model_type.upper()
    
    if model_type == 'NEURAL_NET':
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            predictions = model(X_tensor).cpu().numpy().ravel()
    else:
        predictions = model.predict(X)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['linear', 'ridge', 'lasso', 'xgboost', 'neural_net'],
                      help='Type of model to use for prediction')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file with features')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save predictions')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.input_file}...")
        X = pd.read_csv(args.input_file)
        
        # Preprocess data
        X_preprocessed = preprocess_data(X)
        
        # Load model
        print(f"Loading {args.model_type} model...")
        model = load_model(args.model_type)
        
        # If it's a neural network, set input dimension
        if args.model_type.upper() == 'NEURAL_NET':
            model.input_dim = X.shape[1]
        
        # Make predictions
        print("Making predictions...")
        predictions = predict(model, X_preprocessed, args.model_type)
        
        # Convert predictions back to original scale (if they were log-transformed)
        predictions_original = np.exp(predictions)
        
        # Save predictions
        print(f"Saving predictions to {args.output_file}...")
        results_df = pd.DataFrame({
            'Predicted_DON_log': predictions,
            'Predicted_DON': predictions_original
        })
        
        # Add feature names if they exist in the input file
        if X.columns is not None:
            results_df = pd.concat([X, results_df], axis=1)
        
        results_df.to_csv(args.output_file, index=False)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 