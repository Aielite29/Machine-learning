#!/usr/bin/env python
# coding: utf-8

# In[207]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\kmean_dataset.csv"
data = pd.read_csv(file_path)
X = data.values
X = (X - X.mean()) / X.std()


# In[208]:


def calculate_distortion(data, centroids, labels):
    distortion = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        distortion += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1)**2)
    return distortion


# In[209]:


def kmeans_with_distortion(data, k, max_iterations=10000):
    
    np.random.seed(42)
    initial_centroids = data[np.random.choice(len(data), k, replace=False)]
    centroids = initial_centroids.copy()

    
    for _ in range(max_iterations):
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    
    final_distortion = calculate_distortion(data, centroids, labels)

    return initial_centroids, centroids, labels, final_distortion


# In[210]:


optimal_k = 6
initial_centroids, centroids, labels, final_distortion = kmeans_with_distortion(X, optimal_k)

print("Initial Centroids:")
print(initial_centroids)
print("Final Distortion:", final_distortion)
distortions = []
max_clusters = 178
for k in range(1, max_clusters + 1):
    _, _, _, distortion = kmeans_with_distortion(X, k)
    distortions.append(distortion)


# In[211]:


plt.plot(range(1, max_clusters + 1), distortions, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.show()

plt.scatter(X[:, 0], X[:, 12], c=labels, cmap='viridis', alpha=200)
plt.scatter(centroids[:, 0], centroids[:, 1], c='purple', marker='X', s=200)
plt.title(f'K-means Clustering (k={optimal_k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

data['cluster number'] = labels


# In[212]:


file_path_with_clusters = r"C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\kmean_dataset_with_cluster_number.csv"
data.to_csv(file_path_with_clusters, index=False)


# In[213]:


print(np.array(labels))


# In[ ]:




