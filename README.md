# DON Prediction Model

![corn](corn.jpg)

This project implements various machine learning models for predicting Deoxynivalenol (DON) levels using spectral data. The implementation includes linear regression models with different regularization techniques, gradient boosting, and a neural network approach.

## Project Structure

```
.
├── README.md
├── models /                  # contains the saved models   
├── src/
│   ├── config/
│   │   └── config.py         # Configuration parameters
│   ├── data/
│   │   └── data_loader.py    # Data loading and preprocessing
│   ├── models/
│   │   └── models.py         # Model architectures
│   ├── utils/
│   │   ├── training.py       # Training utilities
│   │   └── visualization.py  # Visualization utilities
|   |   └── model_loading.py  # code to load model
│   └── main.py              # Main script
├── x_processed.csv          # Input features
└── y_processed.csv          # Target values
```

## Notebooks: 

- [Data_exploration_and_preprocessing.ipynb](Data_exploration_and_preprocessing.ipynb): This notebook explores the dataset, and trends & patterns in the dataset. 
- [data_visualization.ipynb](data_visualization.ipynb): Visualizes the processed dataset in 2D 
- [Model_fitting.ipynb](Model_fitting.ipynb): Fitting and evaluating the models on the processed data
- [model_interpretability.ipynb](model_interpretability.ipynb): (incomplete) - Evaluating the model intrepretibility using SHAP

## Project Report: 
- [Project_report.pdf](Project_report.pdf) : Contains detailed report of the project, along with my findings.

## TO-DO 
- [x] Neural network with custom loss
- [ ] Two stage model (classified + Regression)
- [ ] Model interpretibilty 
- [ ] Fast API 
- [ ] Streamlit application 


## Models Implemented

1. Linear Regression
2. Ridge Regression (L2 regularization)
3. Lasso Regression (L1 regularization)
4. XGBoost (Gradient Boosting) (couldn't train this one)
   - Hyperparameter tuning via GridSearchCV
   - Feature importance analysis
   - Multi-core training support
5. Neural Network
   - Multi-layer architecture
   - Dropout regularization
   - Early stopping
6. Neural Network with custom loss 
   - Implemented neural network with weighted MSE loss 
   - Overcame the issues I was getting with regular neural network implementation


## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- XGBoost
- pandas
- numpy
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone (https://github.com/hardikkamboj/DON-prediction-using-hyperspectral-data.git)
cd DON-prediction-using-hyperspectral-data
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Always use the module execution method when running scripts from this project:

```bash

python src/main.py

or 

python -m src.main
```

This will:
1. Load and preprocess the data
2. Train all models (Linear, Ridge, Lasso, Neural Network)
3. Perform hyperparameter tuning
4. Evaluate model performance
5. Generate visualization plots
6. Save the best model

## Model Performance

The code will generate various plots and metrics to compare model performance:
- Training and validation loss curves
- Predicted vs actual value plots
- Model comparison plots (RMSE, R², MAE)
- Feature importance plots for both Lasso and XGBoost


