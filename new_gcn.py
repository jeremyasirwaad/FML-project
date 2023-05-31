import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the Neural Network
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Define the custom dataset
class TabularDataset(Dataset):
    def __init__(self, csv_file, num_rows=10000):
        self.data = pd.read_csv(csv_file, nrows=num_rows)
        self.features = self.data.drop(columns=['label'])
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['label'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.features.iloc[idx].values
        y = self.labels[idx]
        x = torch.Tensor(x)
        y = torch.Tensor([y])  # Assuming y is a single value
        return x, y

# Load the dataset
dataset = TabularDataset('features_reduced.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# Check if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
input_dim = dataset.features.shape[1]
hidden_dim = 64
output_dim = len(np.unique(dataset.labels))  # The number of unique labels in your dataset
model = FFNN(input_dim, hidden_dim, output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels.squeeze().long())
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")

# Evaluation
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        _, predicted = torch.max(output, dim=1)
        total_correct += (predicted == labels.squeeze().long()).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100}%")
