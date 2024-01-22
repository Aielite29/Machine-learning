#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_train.csv'
data = pd.read_csv(file_path)


labels = data.iloc[:, 0]  
features = data.iloc[:, 1:]  


X = features.values
y = labels.values




 





# In[ ]:


def min_max_normalize(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    normalized_X = (X - min_vals) / (max_vals - min_vals + 1e-8)  
    return normalized_X

X_normalized = min_max_normalize(X)


# In[ ]:



np.random.seed(42)
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train = X_normalized[:split_index]
X_test = X_normalized[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]


# In[ ]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_mini_batch(X, y, learning_rate, num_iterations, batch_size):
    num_samples, num_features = X.shape
    weights = np.random.randn(num_features)
    bias = 0

    for i in range(num_iterations):
        for j in range(0, num_samples, batch_size):
            batch_indices = np.random.choice(num_samples, batch_size, replace=False)
            X_batch = X[batch_indices, :]
            y_batch = y[batch_indices]

            linear_model = np.dot(X_batch, weights) + bias
            predictions = sigmoid(linear_model)

            dw = (1 / batch_size) * np.dot(X_batch.T, (predictions - y_batch))
            db = (1 / batch_size) * np.sum(predictions - y_batch)

            weights -= learning_rate * dw
            bias -= learning_rate * db

    return weights, bias


# In[ ]:


def train_one_vs_all_mini_batch(X_train, y_train, learning_rate, num_iterations, batch_size):
    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    classifiers = {}

    for class_value in unique_classes:
        y_binary = np.where(y_train == class_value, 1, 0)

        weights, bias  = logistic_regression_mini_batch(X_train, y_binary, learning_rate, num_iterations, batch_size)
        classifiers[class_value] = (weights, bias)

    return classifiers


# In[ ]:


def predict_one_vs_all(classifiers, X):
    num_samples, _ = X.shape
    num_classes = len(classifiers)
    probabilities = np.zeros((num_samples, num_classes))

    for class_value, (weights, bias) in classifiers.items():
        linear_model = np.dot(X, weights) + bias
        probabilities[:, class_value] = sigmoid(linear_model)

    return np.argmax(probabilities, axis=1)


# In[ ]:


learning_rate = 0.5
num_iterations = 100
batch_size = 64
classifiers = train_one_vs_all_mini_batch(X_train, y_train, learning_rate, num_iterations, batch_size)


y_pred = predict_one_vs_all(classifiers, X_test)

accuracy = np.mean(y_pred == y_test) 
print(f"Accuracy: {accuracy}")



# In[2]:


np.set_printoptions(threshold=np.inf)

test_file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_test.csv'
test_data = pd.read_csv(test_file_path)

test_features = test_data.iloc[:, 1:785].values

X_test_normalized = min_max_normalize(test_features)

test_predictions = predict_one_vs_all(classifiers, X_test_normalized)

print("Predicted Labels:")
print(test_predictions)
    
predictions_df = pd.DataFrame({'Prediction': test_predictions})
predictions_df.to_csv('predictions.csv', index=False)

