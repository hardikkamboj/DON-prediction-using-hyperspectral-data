import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


X = pd.read_csv("x_processed.csv")
y = pd.read_csv("y_processed.csv")

X = np.array(X)
y = np.array(y)

X.shape

y.shape

### Dividing into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training data - ", X_train.shape)
print("Shape of test     data - ", X_test.shape)


# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
    print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
    print(f"Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Actual log(DON)')
    plt.ylabel('Predicted log(DON)')
    plt.title(f'{model_name} - Training Set')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual log(DON)')
    plt.ylabel('Predicted log(DON)')
    plt.title(f'{model_name} - Test Set')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_train': mae_train,
        'mae_test': mae_test
    }

## Linear regression 

linear_model = LinearRegression()
linear_results = evaluate_model(linear_model, X_train, X_test, y_train, y_test, "Simple Linear Regression")

It is overfitting the data

np.logspace(-3, 3, 7)


# Ridge Regression (L2 regularization)
print("\nFitting Ridge Regression...")
# Set up cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Define parameter grid
param_grid = {'alpha': np.logspace(-3, 3, 7)}
# Create and fit grid search
ridge_grid = GridSearchCV(Ridge(random_state=42), param_grid, cv=cv, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
# Get best model
ridge_model = ridge_grid.best_estimator_
print(f"Best Ridge alpha: {ridge_grid.best_params_['alpha']}")
ridge_results = evaluate_model(ridge_model, X_train, X_test, y_train, y_test, "Ridge Regression")

- performs much better than simple linear regression 

print("\nFitting Lasso Regression...")
param_grid = {'alpha': np.logspace(-4, 0, 10)}
lasso_grid = GridSearchCV(Lasso(random_state=42, max_iter=10000, tol=1e-2), param_grid, cv=cv, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
lasso_model = lasso_grid.best_estimator_
print(f"Best Lasso alpha: {lasso_grid.best_params_['alpha']}")
lasso_results = evaluate_model(lasso_model, X_train, X_test, y_train, y_test, "Lasso Regression")

# Count non-zero coefficients (selected features)
n_features = np.sum(lasso_model.coef_ != 0)
print(f"Number of features selected by Lasso: {n_features} out of {X.shape[1]}")

print("\nFitting Neural Network with PyTorch...")

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset for PyTorch
class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.reshape(-1, 1))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets
train_dataset = SpectralDataset(X_train, y_train)
val_dataset = SpectralDataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = self.layer4(x)
        return x

input_dim = X_train.shape[1]
model = NeuralNetwork(input_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200, patience=30):
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
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
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
            
    return model, train_losses, val_losses


# Train the model
model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=200, patience=40
)


# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate PyTorch model
def predict(model, X):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().ravel()
    return y_pred

# Get predictions
y_pred_train_nn = predict(model, X_train)
y_pred_test_nn = predict(model, X_test)


# Calculate metrics
rmse_train_nn = np.sqrt(mean_squared_error(y_train, y_pred_train_nn))
rmse_test_nn = np.sqrt(mean_squared_error(y_test, y_pred_test_nn))
r2_train_nn = r2_score(y_train, y_pred_train_nn)
r2_test_nn = r2_score(y_test, y_pred_test_nn)
mae_train_nn = mean_absolute_error(y_train, y_pred_train_nn)
mae_test_nn = mean_absolute_error(y_test, y_pred_test_nn)

print("\nPyTorch Neural Network Results:")
print(f"Train RMSE: {rmse_train_nn:.4f}, Test RMSE: {rmse_test_nn:.4f}")
print(f"Train R²: {r2_train_nn:.4f}, Test R²: {r2_test_nn:.4f}")
print(f"Train MAE: {mae_train_nn:.4f}, Test MAE: {mae_test_nn:.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_nn, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual log(DON)')
plt.ylabel('Predicted log(DON)')
plt.title('Neural Network - Training Set')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test_nn, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual log(DON)')
plt.ylabel('Predicted log(DON)')
plt.title('Neural Network - Test Set')
plt.tight_layout()
plt.show()


# 7. Summary of all models
models = [
    ('Simple Linear Regression', linear_results),
    ('Ridge Regression', ridge_results),
    ('Lasso Regression', lasso_results),
    ('PyTorch Neural Network', {
        'rmse_train': rmse_train_nn,
        'rmse_test': rmse_test_nn,
        'r2_train': r2_train_nn,
        'r2_test': r2_test_nn,
        'mae_train': mae_train_nn,
        'mae_test': mae_test_nn
    })
]


comparison_df = pd.DataFrame({
    'Model': [name for name, _ in models],
    'Train RMSE': [results['rmse_train'] for _, results in models],
    'Test RMSE': [results['rmse_test'] for _, results in models],
    'Train R²': [results['r2_train'] for _, results in models],
    'Test R²': [results['r2_test'] for _, results in models],
    'Train MAE': [results['mae_train'] for _, results in models],
    'Test MAE': [results['mae_test'] for _, results in models]
})

# Plot comparison (continued)
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


# 8. Converting predictions back to original scale
# For the best model (let's assume it's the neural network)
y_test_original = np.exp(y_test)
y_pred_test_nn_original = np.exp(y_pred_test_nn)

# Calculate metrics in original scale
rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_test_nn_original))
r2_original = r2_score(y_test_original, y_pred_test_nn_original)
mae_original = mean_absolute_error(y_test_original, y_pred_test_nn_original)

print("\nNeural Network Results in Original Scale:")
print(f"RMSE: {rmse_original:.4f}")
print(f"R²: {r2_original:.4f}")
print(f"MAE: {mae_original:.4f}")

# Plot in original scale
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_test_nn_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
plt.xlabel('Actual DON')
plt.ylabel('Predicted DON')
plt.title('Neural Network Predictions in Original Scale')
plt.tight_layout()
plt.show()


if n_features > 0:  # If Lasso selected any features
    # Get non-zero coefficients and their corresponding feature indices
    non_zero_indices = np.where(lasso_model.coef_ != 0)[0]
    non_zero_coefs = lasso_model.coef_[non_zero_indices]
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature Index': non_zero_indices,
        'Coefficient': non_zero_coefs
    })
    
    # Sort by absolute coefficient value
    feature_importance_df['Abs Coefficient'] = np.abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.sort_values('Abs Coefficient', ascending=False)
    
    # Plot top features (up to 10)
    top_n = min(10, len(feature_importance_df))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature Index', 
                data=feature_importance_df.head(top_n),
                palette='coolwarm')
    plt.title(f'Top {top_n} Important Wavelengths Selected by Lasso')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Wavelength Index')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Important Wavelengths:")
    print(feature_importance_df.head(10))





# 9. Feature importance analysis for Lasso (since it performs feature selection)


# 10. Save the best model (assuming it's the neural network)
torch.save(model.state_dict(), 'best_hyperspectral_model.pth')
print("\nBest model saved as 'best_hyperspectral_model.pth'")

# Function to load the model for future use
def load_model(model_path, input_dim):
    loaded_model = NeuralNetwork(input_dim)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    return loaded_model

# Example of how to use the model for prediction on new data:
'''
# Load the model
loaded_model = load_model('best_hyperspectral_model.pth', X.shape[1])

# Preprocess new data (assuming it's already normalized like the training data)
# new_data = ...

# Make predictions
new_data_tensor = torch.FloatTensor(new_data).to(device)
with torch.no_grad():
    predictions_log = loaded_model(new_data_tensor).cpu().numpy().ravel()
    
# Convert back to original scale
predictions_original = np.exp(predictions_log)
'''

# 11. Cross-Validation for the Neural Network
print("\nPerforming 5-fold Cross-Validation for Neural Network...")

def cross_validate_nn(X, y, n_splits=5, epochs=100, patience=15):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create datasets and loaders
        train_dataset = SpectralDataset(X_train_fold, y_train_fold)
        val_dataset = SpectralDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        fold_model = NeuralNetwork(input_dim).to(device)
        optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
        
        # Train model
        fold_model, _, _ = train_model(
            fold_model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=epochs, patience=patience
        )
        
        # Evaluate
        y_pred_fold = predict(fold_model, X_val_fold)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        r2 = r2_score(y_val_fold, y_pred_fold)
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        
        fold_results.append({
            'fold': fold+1,
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        })
        
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    
    # Summarize results
    df_cv = pd.DataFrame(fold_results)
    print("\nCross-Validation Summary:")
    print(f"Mean RMSE: {df_cv['rmse'].mean():.4f} ± {df_cv['rmse'].std():.4f}")
    print(f"Mean R²: {df_cv['r2'].mean():.4f} ± {df_cv['r2'].std():.4f}")
    print(f"Mean MAE: {df_cv['mae'].mean():.4f} ± {df_cv['mae'].std():.4f}")
    
    return df_cv

# Run cross-validation (with fewer epochs for demonstration)
cv_results = cross_validate_nn(X, y, n_splits=5, epochs=50, patience=10)

# Plot cross-validation results
plt.figure(figsize=(10, 6))
sns.boxplot(x='variable', y='value', data=pd.melt(cv_results[['rmse', 'r2', 'mae']], 
                                                var_name='variable', value_name='value'))
plt.title('Neural Network Cross-Validation Results')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.show()2