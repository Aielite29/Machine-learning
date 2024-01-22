#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_train.csv'
data = pd.read_csv(file_path)


X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values  


X = (X - X.mean()) / X.std()

num_classes = 10
y_one_hot = np.eye(num_classes)[y]

split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_cv = X[:split_index], X[split_index:]
y_train, y_cv = y_one_hot[:split_index], y_one_hot[split_index:]


# In[56]:


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * np.sqrt(2 / layer_dims[l - 1])
        parameters[f'b{l}'] = np.zeros((1, layer_dims[l]))
    return parameters


# In[57]:


def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        Z = np.dot(A, parameters[f'W{l}']) + parameters[f'b{l}']
        A = relu(Z)
        caches.append((Z, A))
    
    ZL = np.dot(A, parameters[f'W{L}']) + parameters[f'b{L}']
    AL = softmax(ZL)
    caches.append((ZL, AL))
    
    return AL, caches


# In[58]:


def compute_cost(AL, y):
    m = y.shape[0]
    cost = -np.sum(np.multiply(y, np.log(AL + 1e-8))) / m
    return cost


# In[59]:


def backward_propagation(AL, y, caches, parameters):
    grads = {}
    L = len(caches)
    m = AL.shape[0]
    dAL = AL - y
    
    current_cache = caches[L - 1]
    ZL, AL = current_cache
    dZL = dAL
    
    grads[f'dW{L}'] = np.dot(caches[L - 2][1].T, dZL) / m
    grads[f'db{L}'] = np.sum(dZL, axis=0, keepdims=True) / m
    
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        Z, A = current_cache
        
        dA = np.dot(dZL, parameters[f'W{L}'].T)
        dZ = np.multiply(dA, np.int64(A > 0))
        
        grads[f'dW{l + 1}'] = np.dot(caches[l - 1][1].T, dZ) / m if l > 0 else np.dot(X_train.T, dZ) / m

        grads[f'db{l + 1}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        dZL = dZ
        L -= 1
    
    return grads


# In[60]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters[f'W{l + 1}'] -= learning_rate * grads[f'dW{l + 1}']
        parameters[f'b{l + 1}'] -= learning_rate * grads[f'db{l + 1}']
    
    return parameters


# In[61]:


def calculate_accuracy(X_data, y_data, parameters):
    AL, _ = forward_propagation(X_data, parameters)
    predictions = np.argmax(AL, axis=1)
    ground_truth = np.argmax(y_data, axis=1)
    accuracy = np.mean(predictions == ground_truth)
    return accuracy


# In[69]:


def neural_network_model(X_train, y_train, X_cv, y_cv, num_hidden_layers, num_classes, learning_rate, num_iterations):
    input_layer_size = X_train.shape[1]
    output_layer_size = num_classes
    
    layer_dims = [input_layer_size] + [32 * (2 ** max(0, num_hidden_layers - i - 1)) for i in range(num_hidden_layers)] + [output_layer_size]
    
    train_costs = []
    cv_costs = []
    train_accuracies = []
    cv_accuracies = []
    parameters = initialize_parameters(layer_dims)
    
    for i in range(num_iterations):
        AL_train, caches = forward_propagation(X_train, parameters)
        cost_train = compute_cost(AL_train, y_train)
        grads = backward_propagation(AL_train, y_train, caches, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            accuracy_train = calculate_accuracy(X_train, y_train, parameters)
            accuracy_cv = calculate_accuracy(X_cv, y_cv, parameters)
            train_accuracies.append(accuracy_train)
            cv_accuracies.append(accuracy_cv)
            print(f'Iteration {i}: Train Cost: {cost_train}, Train Accuracy: {accuracy_train}, CV Accuracy: {accuracy_cv}')
        
        train_costs.append(cost_train)
    
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.squeeze(train_costs), label='Train')
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title(f'Learning curve: Train vs CV')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(cv_accuracies, label='CV')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations (per hundreds)')
    plt.title(f'Accuracy during Training and CV')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return parameters, cv_accuracies


# In[70]:


num_hidden_layers = 4
learning_rate = 0.1
num_iterations = 5000
trained_parameters,cv_accuracies = neural_network_model(X_train, y_train, X_cv, y_cv, num_hidden_layers, num_classes, learning_rate, num_iterations)


# In[133]:


file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_test.csv'
data = pd.read_csv(file_path)
X_test = data.iloc[:, 1:785].values 

def predict_labels(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predicted_labels = np.argmax(AL, axis=1)
    return predicted_labels


predicted_labels = predict_labels(X_test, trained_parameters)
np.set_printoptions(threshold=np.inf)
print(list(predicted_labels))


# In[134]:


data['predicted_labels'] = predicted_labels

file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Classification_test_with_predictions.csv'
data.to_csv(file_path, index=False)


# In[ ]:




