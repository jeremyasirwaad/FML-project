import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# Define the Graph Convolutional Layer
class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        x = torch.matmul(adjacency_matrix, x.unsqueeze(-1)).squeeze(-1)
        x = self.linear(x)
        return x

# Define the Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.layer2 = GraphConvolutionLayer(hidden_dim, output_dim)
        
    def forward(self, x, adjacency_matrix):
        x = torch.relu(self.layer1(x, adjacency_matrix))
        x = self.layer2(x, adjacency_matrix)
        return x

# Define the custom dataset
class GraphDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.drop(columns=['label'])
        self.adjacency_matrix = self._calculate_adjacency_matrix()
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['label'])  # Encode labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.features.iloc[idx].values
        y = self.labels[idx]
        adj = self.adjacency_matrix[idx]
        x = torch.Tensor(x)
        y = torch.Tensor([y])  # Assuming y is a single value
        adj = torch.Tensor(adj)
        return x, y, adj
    
    def _calculate_adjacency_matrix(self):
        num_nodes = len(self.data)
        adjacency_matrix = np.ones((num_nodes, num_nodes))
        return adjacency_matrix


# Load the dataset
dataset = GraphDataset('./j.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# Check if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the GCN model
input_dim = dataset.features.shape[1]
hidden_dim = 64
output_dim = 2  # Assuming you have two classes: 'benign' and 'malicious'
model = GCN(input_dim, hidden_dim, output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels, adjacency_matrix) in enumerate(train_loader):
        features, labels, adjacency_matrix = features.to(device), labels.to(device), adjacency_matrix.to(device)
        
        optimizer.zero_grad()
        output = model(features, adjacency_matrix)
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
    for features, labels, adjacency_matrix in test_loader:
        features, labels, adjacency_matrix = features.to(device), labels.to(device), adjacency_matrix.to(device)
        output = model(features, adjacency_matrix)
        _, predicted = torch.max(output, dim=1)
        total_correct += (predicted == labels.squeeze().long()).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100}%")
