import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Select the features for clustering
features = data[['url_length', 'no_of_digits', 'no_of_parameters', 'has_port', 'url_path_length', 'is_https', 'no_of_sub_domains', 'url_entropy', 'no_of_special_chars', 'contains_IP', 'no_of_subdir', 'url_is_encoded', 'domain_length', 'no_of_queries', 'avg_token_length', 'token_count', 'largest_token', 'smallest_token', 'contains_at_symbol', 'is_shortened', 'count_dots', 'count_delimiters', 'count_sub_domains', 'is_www', 'count_reserved_chars']]

# Perform K-means clustering
k = 2 # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(features)

# Assign cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Visualize clusters using scatter plot
plt.scatter(data['url_length'], data['no_of_digits'], c=data['Cluster'])
plt.xlabel('URL Length')
plt.ylabel('Number of Digits')
plt.title('URL Clusters')
plt.show()
