#!/usr/bin/env python
# coding: utf-8

# In[246]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Lineardata_train.csv'
data = pd.read_csv(file_path)


X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values   


# In[247]:


def Z_Score_Normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

X_normalized, mean_X, std_X = Z_Score_Normalization(X)

print(X_normalized)
X=X_normalized


# In[266]:


alpha = 0.9
iterations = 10000
num_features = X_scaled.shape[1]
np.random.seed(42)
weights = np.random.randn(num_features)
b=np.random.randn(1)


# In[267]:


print(f"Weights : {weights[:]}")
print(f"Bias : {b}")


# In[268]:


def cost(X, y, weights,b):
    m = len(y)
    predictions = np.dot(X, weights) + b
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost
z= cost(X, y, weights,b)


# In[269]:


def gradient_descent(X, y, weights, alpha, iterations,b):
    m = len(y)
    
    cost_history = []
    for i in range(iterations):
        predictions = np.dot(X, weights)+b
        error = predictions - y
        gradient_w = (1 / m) * np.dot(X.T, error)
        gradient_b= (1 / m) *np.sum(error)
        
        weights -= alpha * gradient_w
        b -= alpha * gradient_b
        costs = cost(X, y, weights,b)
        cost_history.append(costs)
        
    return weights, cost_history,b
optimal_weights, cost_history,b = gradient_descent(X_normalized, y, weights, alpha, iterations,b)


# In[270]:


print(f"starting cost : {cost_history[0]}")
print(f"Final cost: {cost_history[-1]}")
print(f"Weights : {optimal_weights[:]}")
print(f"Bias : {b}")


# In[271]:


plt.figure()
plt.plot(range(iterations), cost_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Number of Iterations')
plt.show()


# In[272]:


def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    
    ss_residual = np.sum((y_true - y_pred)**2)
    
    r2 = 1 - (ss_residual / ss_total)
    
    return r2

predicted_y = np.dot(X_scaled, optimal_weights) + b

r2 = r2_score(y, predicted_y)

print(f"R2 Score: {r2}")


# In[273]:


plt.figure()
plt.plot(X[:,19], y)
plt.xlabel('Feature 20')

plt.ylabel('target value ')
plt.title('feature 20 vs. target ')
plt.show()


# In[274]:


print(cost_history)


# In[288]:


np.set_printoptions(threshold=np.inf)
test_file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Lineardata_test.csv'
test_data = pd.read_csv(test_file_path)

X_test = test_data.iloc[:, 1:].values 
X_test_normalised = (X_test - mean_X) / std_X
predictions_test = np.dot(X_test_normalised, optimal_weights) + b

x = list(predictions_test)

print("predicted list")
print(x)


# In[290]:


data['predicted_labels'] = x


data.to_csv(test_file_path, index=False)


# In[ ]:




