#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_train.csv'
data = pd.read_csv(file_path)

labels = data.iloc[:, 0].values  
features = data.iloc[:, 1:].values  


# In[3]:


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# In[4]:


np.random.seed(42)  
train_size = int(0.8 * len(features))
indices = np.random.permutation(len(features))
train_indices, test_indices = indices[:train_size], indices[train_size:]
X_train, X_test = features[train_indices], features[test_indices]
y_train, y_test = labels[train_indices], labels[test_indices]


# In[12]:


def precalculate_distances(X_train, X_test):
    distances = np.zeros((len(X_test), len(X_train)))
    for i, test_point in enumerate(X_test):
        for j, train_point in enumerate(X_train):
            distances[i, j] = euclidean_distance(test_point, train_point)
    return distances


distances = precalculate_distances(X_train, X_test)


# In[13]:


def predict_with_distances(distances, y_train, k):
    predictions = []
    for i in range(len(distances)):
        sorted_indices = np.argsort(distances[i])[:k]
        k_nearest_labels = [y_train[idx] for idx in sorted_indices]
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(prediction)
    return predictions

k_values = range(1, 40)
accuracies = []

for k_value in k_values:
    predicted_labels = predict_with_distances(distances, y_train, k_value)
    accuracy = np.mean(predicted_labels == y_test)
    accuracies.append(accuracy)


# In[14]:


plt.figure(figsize=(12, 8))
plt.plot(k_values, accuracies, marker='x', linestyle='-')
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[ ]:


test_file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_test.csv'
test_data = pd.read_csv(test_file_path)

test_ids = test_data.iloc[:, 0].values
test_features = test_data.iloc[:, 1:785].values

test_distances = precalculate_distances(features, test_features)
k_value = 1

all_predicted_labels = predict_with_distances(test_distances, labels, k_value)


print("List of predicted labels for all examples:")
np.set_printoptions(threshold=np.inf)
print(list(all_predicted_labels))


# In[ ]:


print("List of predicted labels for all examples:")
np.set_printoptions(threshold=np.inf)
print(list(all_predicted_labels))


# In[ ]:




