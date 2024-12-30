import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

# Load data
train_data = pd.read_csv('d:/code/MachineLearning/houseprice/train.csv')
test_data = pd.read_csv('d:/code/MachineLearning/houseprice/test.csv')

# Preprocess data
def preprocess_data(data):
    # Fill missing values
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna('Missing')
        else:
            data[column] = data[column].fillna(data[column].median())
    
    # Feature engineering
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    # Select numerical and categorical features
    num_features = data.select_dtypes(include=[np.number]).columns
    cat_features = data.select_dtypes(include=['object']).columns

    # One-hot encode categorical variables and scale numerical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    return preprocessor.fit_transform(data)

# Apply preprocessing
X = preprocess_data(train_data.drop(['Id', 'SalePrice'], axis=1))
X_test = preprocess_data(test_data.drop('Id', axis=1))
y = np.log1p(train_data['SalePrice'])  # Log transform target

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Define model
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Model, optimizer, and loss function
input_dim = X.shape[1]
model = HousePriceModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training with K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = torch.sqrt(criterion(outputs, batch_y))  # RMSE loss
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = torch.sqrt(criterion(val_outputs, y_val))
        print(f'Fold {fold+1}, Validation RMSE: {val_loss.item():.4f}')

# Final predictions on test set
model.eval()
with torch.no_grad():
    predictions = model(X_test).exp().detach().numpy()  # Inverse of log1p

# Save submission
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions.flatten()})
submission.to_csv('submission_3.csv', index=False)
