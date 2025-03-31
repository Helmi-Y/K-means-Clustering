import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Load the data from the text file
data = pd.read_csv('iris_bezdek.txt', delim_whitespace=True, header=None)

# Function to calculate the Dunn index


def dunn_index(X, labels):
    X = np.array(X)  # Convert data to numpy array
    clusters = [X[labels == i] for i in np.unique(labels)]
    inter_cluster_distances = []
    intra_cluster_distances = []

    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i < j:
                inter_cluster_distances.append(
                    np.min(cdist(cluster1, cluster2)))

        # Calculate intra-cluster distance (max pairwise distance in the cluster)
        intra_cluster_distances.append(
            np.max(np.linalg.norm(cluster1 - cluster1[:, np.newaxis], axis=2)))

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)


# Range of k values to try
k_range = range(2, 9)

# Store the scores for each k
results = {
    'k': [],
    'Calinski-Harabasz': [],
    'Silhouette Width': [],
    'Dunn Index': [],
    'Davies-Bouldin': []
}

for k in k_range:
    # Apply k-means clustering with random initialization
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Compute the evaluation metrics
    ch_score = calinski_harabasz_score(data, labels)
    sw_score = silhouette_score(data, labels)
    dunn_index_value = dunn_index(data, labels)
    db_score = davies_bouldin_score(data, labels)

    # Store the results
    results['k'].append(k)
    results['Calinski-Harabasz'].append(ch_score)
    results['Silhouette Width'].append(sw_score)
    results['Dunn Index'].append(dunn_index_value)
    results['Davies-Bouldin'].append(db_score)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)
