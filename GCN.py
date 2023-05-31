import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from nltk.tokenize import word_tokenize
from scipy.stats import entropy



from urllib.parse import urlparse

import re

SPECIAL_CHARS = "!@#$%^&*()-_=+[{]}\|;:'\",<.>/?"
IP_REGEX = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load and preprocess your dataset from the CSV file
# Assuming the dataset is stored in a CSV file named 'dataset.csv'

# Read the CSV file and extract the features and labels
import pandas as pd

df = pd.read_csv('./features2.csv')

urls = df['url'].tolist()
labels = df['label'].tolist()

# Preprocess and convert features to tensor
# Modify this section based on your specific preprocessing steps
url_length = df['url_length'].tolist()
no_of_digits = df['no_of_digits'].tolist()
no_of_parameters = df['no_of_parameters'].tolist()
has_port = df['has_port'].tolist()
url_path_length = df['url_path_length'].tolist()
is_https = df['is_https'].tolist()
no_of_sub_domains = df['no_of_sub_domains'].tolist()
url_entropy = df['url_entropy'].tolist()
no_of_special_chars = df['no_of_special_chars'].tolist()
contains_IP = df['contains_IP'].tolist()
no_of_subdir = df['no_of_subdir'].tolist()
url_is_encoded = df['url_is_encoded'].tolist()
domain_length = df['domain_length'].tolist()
no_of_queries = df['no_of_queries'].tolist()
avg_token_length = df['avg_token_length'].tolist()
token_count = df['token_count'].tolist()
largest_token = df['largest_token'].tolist()
smallest_token = df['smallest_token'].tolist()
contains_at_symbol = df['contains_at_symbol'].tolist()
is_shortened = df['is_shortened'].tolist()
count_dots = df['count_dots'].tolist()
count_delimiters = df['count_delimiters'].tolist()
count_sub_domains = df['count_sub_domains'].tolist()
is_www = df['is_www'].tolist()
count_reserved_chars = df['count_reserved_chars'].tolist()

num_urls = len(urls)
edge_index = torch.tensor([[i, j] for i in range(num_urls) for j in range(num_urls) if i != j], dtype=torch.long).t()
# edge_index = torch.tensor(list(edges), dtype=torch.long).t()


# Combine features into a tensor
x = torch.tensor([
    url_length, no_of_digits, no_of_parameters, has_port, url_path_length,
    is_https, no_of_sub_domains, url_entropy, no_of_special_chars, contains_IP,
    no_of_subdir, url_is_encoded, domain_length, no_of_queries, avg_token_length,
    token_count, largest_token, smallest_token, contains_at_symbol, is_shortened,
    count_dots, count_delimiters, count_sub_domains, is_www, count_reserved_chars
], dtype=torch.float).t()

# Convert labels to tensor
label_mapping = {'benign': 0, 'malicious': 1}  # Modify based on your label mapping
y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)

# Create the edge_index tensor
num_urls = len(urls)
edges = []
for i in range(num_urls):
    for j in range(num_urls):
        if i != j:
            edges.append([i, j])
edges = torch.tensor(edges, dtype=torch.long).t()

# Create the Data object
data = Data(x=x, edge_index=edge_index, y=y)

# Set the number of features and classes based on your dataset
num_features = data.num_features
num_classes = 2  # Assuming binary classification (benign or malicious)

# Initialize the model and optimizer
model = GCN(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


import random

# Shuffle the indices
indices = list(range(len(urls)))
random.shuffle(indices)

# Set the ratio for train-test split
train_ratio = 0.8

# Split the indices into train and test sets
train_size = int(train_ratio * len(urls))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create the train mask
train_mask = torch.zeros(len(urls), dtype=torch.bool)
train_mask[train_indices] = True

# Create the test mask (optional)
test_mask = torch.zeros(len(urls), dtype=torch.bool)
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask

num_epochs = 10
# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    accuracy = correct.sum().item() / data.test_mask.sum().item()
    print('Accuracy: {:.2f}'.format(accuracy) * 100)

# # Assuming the user input is stored in a variable called 'user_input'
# user_input = "https://www.example.com"

# # Preprocess the user input to extract the required features
# new_url = user_input
# new_url_length = len(new_url)
# new_no_of_digits = sum(char.isdigit() for char in new_url)
# new_no_of_parameters = new_url.count('?')
# new_has_port = urlparse(new_url).port != None
# new_url_path_length = len(urlparse(new_url).path)
# new_is_https = urlparse(new_url).scheme == 'https'
# new_no_of_sub_domains = len(urlparse(new_url).hostname.split('.')[:-2])
# new_url_entropy = entropy(bytearray(new_url, 'utf-8'))
# new_no_of_special_chars = sum(char in SPECIAL_CHARS for char in new_url)
# new_contains_IP = 1 if re.match(IP_REGEX, new_url) else 0
# # ... Add preprocessing steps for other features ...

# # Convert the features to tensors
# new_x = torch.tensor([
#     new_url_length, new_no_of_digits, new_no_of_parameters, new_has_port, new_url_path_length,
#     new_is_https, new_no_of_sub_domains, new_url_entropy, new_no_of_special_chars, new_contains_IP,
#     # ... Add other features ...
# ], dtype=torch.float).unsqueeze(0)

# # Pass the features through the trained model
# model.eval()
# with torch.no_grad():
#     new_out = model(new_x, data.edge_index)
#     new_pred = new_out.argmax(dim=1)

# # Convert the predicted class to the corresponding label
# label_mapping = {0: 'benign', 1: 'malicious'}  # Modify based on your label mapping
# predicted_label = label_mapping[new_pred.item()]

# print("Predicted Label:", predicted_label)

