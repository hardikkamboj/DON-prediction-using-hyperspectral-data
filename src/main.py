"""
Main script for training and evaluating DON prediction models.
"""

import torch
import joblib
from src.data.data_loader import load_data, create_data_splits, create_data_loaders
from src.models.models import NeuralNetwork, LinearModels, XGBoostModel
from src.utils.training import (
    train_neural_network, train_neural_network_weighted,
    evaluate_model, predict, optimize_hyperparameters,
    train_neural_network_with_optimal_params
)
from src.utils.visualization import (
    plot_training_history, plot_predictions, plot_model_comparison,
    plot_feature_importance
)
from src.config.config import DEVICE, MODEL_PATHS
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    """Main function to run the DON prediction pipeline."""
    
    # Load and prepare data
    print("Loading data...")
    X, y = load_data()
    
    # Scale the data
    # print("Scaling data...")
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    print("Saving scaler...")
    # joblib.dump(scaler, MODEL_PATHS['SCALER'])
    
    # Split the data
    X_train, X_test, y_train, y_test = create_data_splits(X, y)
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    # Initialize models
    print("\nInitializing models...")
    linear_model = LinearModels.get_linear_regression()
    ridge_model = LinearModels.get_ridge_regression()
    lasso_model = LinearModels.get_lasso_regression()
    xgb_model = XGBoostModel.get_model()
    nn_model = NeuralNetwork(X.shape[1]).to(DEVICE)
    
    # Train and evaluate linear models
    print("\nTraining and evaluating linear models...")
    models_results = []
    
    # Linear Regression
    linear_model.fit(X_train, y_train)
    linear_metrics = evaluate_model(linear_model, X_train, X_test, y_train, y_test)
    models_results.append(("Linear Regression", linear_metrics))
    plot_predictions(y_train, predict(linear_model, X_train), 
                    y_test, predict(linear_model, X_test), 
                    "Linear Regression")
    # Save linear model
    joblib.dump(linear_model, MODEL_PATHS['LINEAR'])
    print("Linear model saved!")
    
    # Ridge Regression
    print("\nTraining Ridge Regression...")
    ridge_model.fit(X_train, y_train)
    print(f"Best Ridge alpha: {ridge_model.best_params_['alpha']}")
    ridge_metrics = evaluate_model(ridge_model.best_estimator_, X_train, X_test, y_train, y_test)
    models_results.append(("Ridge Regression", ridge_metrics))
    plot_predictions(y_train, predict(ridge_model.best_estimator_, X_train),
                    y_test, predict(ridge_model.best_estimator_, X_test),
                    "Ridge Regression")
    # Save ridge model
    joblib.dump(ridge_model.best_estimator_, MODEL_PATHS['RIDGE'])
    print("Ridge model saved!")
    
    # Lasso Regression
    print("\nTraining Lasso Regression...")
    lasso_model.fit(X_train, y_train)
    print(f"Best Lasso alpha: {lasso_model.best_params_['alpha']}")
    lasso_metrics = evaluate_model(lasso_model.best_estimator_, X_train, X_test, y_train, y_test)
    models_results.append(("Lasso Regression", lasso_metrics))
    plot_predictions(y_train, predict(lasso_model.best_estimator_, X_train),
                    y_test, predict(lasso_model.best_estimator_, X_test),
                    "Lasso Regression")
    # Save lasso model
    joblib.dump(lasso_model.best_estimator_, MODEL_PATHS['LASSO'])
    print("Lasso model saved!")
    
    # Plot Lasso feature importance
    non_zero_coef = lasso_model.best_estimator_.coef_ != 0
    if non_zero_coef.any():
        feature_indices = np.where(non_zero_coef)[0]
        coefficients = lasso_model.best_estimator_.coef_[non_zero_coef]
        plot_feature_importance(feature_indices, coefficients)
    
    # XGBoost
    # print("\nTraining XGBoost...")
    # xgb_model.fit(X_train, y_train)
    # print("\nBest XGBoost parameters:")
    # for param, value in xgb_model.best_params_.items():
    #     print(f"{param}: {value}")
    # xgb_metrics = evaluate_model(xgb_model.best_estimator_, X_train, X_test, y_train, y_test)
    # models_results.append(("XGBoost", xgb_metrics))
    # plot_predictions(y_train, predict(xgb_model.best_estimator_, X_train),
    #                 y_test, predict(xgb_model.best_estimator_, X_test),
    #                 "XGBoost")
    # # Save XGBoost model
    # joblib.dump(xgb_model.best_estimator_, MODEL_PATHS['XGBOOST'])
    # print("XGBoost model saved!")
    
    # # Plot XGBoost feature importance
    # feature_importance = xgb_model.best_estimator_.feature_importances_
    # feature_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
    # plot_feature_importance(feature_indices, feature_importance[feature_indices], top_n=10)
    
    # Train and evaluate regular neural network with hyperparameter optimization
    print("\nOptimizing Regular Neural Network Hyperparameters...")
    regular_nn_params = optimize_hyperparameters(
        NeuralNetwork, 
        train_loader, 
        test_loader,
        n_trials=50  # Adjust number of trials as needed
    )
    print("\nBest hyperparameters for Regular Neural Network:")
    for param, value in regular_nn_params.items():
        print(f"{param}: {value}")
    
    print("\nTraining Regular Neural Network with optimal parameters...")
    nn_model, history = train_neural_network_with_optimal_params(
        NeuralNetwork,
        train_loader,
        test_loader,
        regular_nn_params
    )
    plot_training_history(history)
    
    nn_metrics = evaluate_model(nn_model, X_train, X_test, y_train, y_test, is_neural_net=True)
    models_results.append(("Neural Network (Optimized)", nn_metrics))
    plot_predictions(y_train, predict(nn_model, X_train, is_neural_net=True),
                    y_test, predict(nn_model, X_test, is_neural_net=True),
                    "Neural Network (Optimized)")
    
    # Train and evaluate weighted neural network with hyperparameter optimization
    print("\nOptimizing Weighted Neural Network Hyperparameters...")
    weighted_nn_params = optimize_hyperparameters(
        NeuralNetwork,
        train_loader,
        test_loader,
        is_weighted=True,
        weight_threshold=1.0,  # Adjust based on your data
        n_trials=50  # Adjust number of trials as needed
    )
    print("\nBest hyperparameters for Weighted Neural Network:")
    for param, value in weighted_nn_params.items():
        print(f"{param}: {value}")
    
    print("\nTraining Weighted Neural Network with optimal parameters...")
    weighted_nn_model, weighted_history = train_neural_network_with_optimal_params(
        NeuralNetwork,
        train_loader,
        test_loader,
        weighted_nn_params,
        is_weighted=True,
        weight_threshold=1.0  # Same threshold as used in optimization
    )
    plot_training_history(weighted_history)
    
    weighted_nn_metrics = evaluate_model(weighted_nn_model, X_train, X_test, y_train, y_test, is_neural_net=True)
    models_results.append(("Weighted Neural Network (Optimized)", weighted_nn_metrics))
    plot_predictions(y_train, predict(weighted_nn_model, X_train, is_neural_net=True),
                    y_test, predict(weighted_nn_model, X_test, is_neural_net=True),
                    "Weighted Neural Network (Optimized)")
    
    # Compare all models
    plot_model_comparison(models_results)
    
    # Save neural networks with their optimal parameters
    model_info = {
        'state_dict': nn_model.state_dict(),
        'hyperparameters': regular_nn_params
    }
    torch.save(model_info, MODEL_PATHS['NEURAL_NET'])
    print("Regular Neural Network saved with hyperparameters!")
    
    weighted_model_info = {
        'state_dict': weighted_nn_model.state_dict(),
        'hyperparameters': weighted_nn_params
    }
    weighted_model_path = MODEL_PATHS['NEURAL_NET'].replace('.pth', '_weighted.pth')
    torch.save(weighted_model_info, weighted_model_path)
    print("Weighted Neural Network saved with hyperparameters!")

if __name__ == "__main__":
    main() 