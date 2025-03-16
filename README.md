# DON Prediction Model

This project implements various machine learning models for predicting Deoxynivalenol (DON) levels using spectral data. The implementation includes linear regression models with different regularization techniques, gradient boosting, and a neural network approach.

## Project Structure

```
.
├── README.md
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
│   └── main.py              # Main script
├── x_processed.csv          # Input features
└── y_processed.csv          # Target values
```

## Models Implemented

1. Linear Regression
2. Ridge Regression (L2 regularization)
3. Lasso Regression (L1 regularization)
4. XGBoost (Gradient Boosting)
   - Hyperparameter tuning via GridSearchCV
   - Feature importance analysis
   - Multi-core training support
5. Neural Network
   - Multi-layer architecture
   - Dropout regularization
   - Early stopping

## Features

### Model Training
- Automated hyperparameter tuning for Ridge, Lasso, and XGBoost models
- Cross-validation for robust performance estimation
- Early stopping for Neural Network to prevent overfitting
- Multi-core processing support for XGBoost

### Model Evaluation
- Comprehensive metrics (RMSE, R², MAE)
- Training and validation loss curves
- Predicted vs actual value plots
- Model comparison visualizations

### Feature Analysis
- Lasso regression feature selection
- XGBoost feature importance ranking
- Visualization of top contributing features

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

## Running Scripts Properly

When running scripts from this project, it's important to use the correct Python command to ensure all imports work properly. There are two ways to run Python scripts, and they behave differently:

1. **Direct script execution** (`python src/main.py`):
   - This method directly executes the script file
   - Can cause import errors because Python doesn't recognize the package structure
   - Not recommended for this project

2. **Module execution** (`python -m src.main`):
   - This is the **recommended** way to run scripts in this project
   - Properly recognizes the `src` package structure
   - Correctly resolves all imports within the package
   - Must be run from the project root directory

Always use the module execution method when running scripts from this project:

```bash

python src/main.py

or 

python -m src.main
```

## Usage

Run the main script to train and evaluate all models:

```bash
python -m src.main
```

This will:
1. Load and preprocess the data
2. Train all models (Linear, Ridge, Lasso, XGBoost, Neural Network)
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

## Model Configuration

### XGBoost Parameters
The XGBoost model can be configured in `config.py` with the following parameters:
- max_depth: Controls tree depth
- learning_rate: Step size shrinkage
- n_estimators: Number of boosting rounds
- min_child_weight: Minimum sum of instance weight
- subsample: Subsample ratio of training instances
- colsample_bytree: Subsample ratio of columns
- gamma: Minimum loss reduction for split

### Neural Network Parameters
The Neural Network can be configured with:
- Hidden layer sizes
- Dropout rate
- Learning rate
- Number of epochs
- Early stopping patience

## Model Saving and Loading

The best model (Neural Network) is automatically saved. To load and use the saved model:

```python
from src.models.models import NeuralNetwork
from src.config.config import MODEL_PATH

# Initialize model
model = NeuralNetwork(input_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Make predictions
predictions = model(input_data)
```

## Contributing

Feel free to submit issues and enhancement requests. Some areas for potential improvement:
- Additional model architectures
- Advanced preprocessing techniques
- Hyperparameter optimization strategies
- Ensemble methods
- Model interpretability tools