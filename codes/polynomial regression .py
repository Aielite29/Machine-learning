#!/usr/bin/env python
# coding: utf-8

# In[144]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Polynomialdata_train.csv'
data = pd.read_csv(file_path)

X = data[['feature 1', 'feature 2', 'feature 3']].values
y = data['target'].values


# In[145]:


def create_polynomial_features_3vars(X, degree):
    
    feature_values = X[:, :3]  
    
    n_samples, n_features = feature_values.shape
    poly_features = np.ones((n_samples, 1))  
    
    for d in range(1, degree + 1):
        exponents_combinations = np.array([[a, b, c] for a in range(d + 1) for b in range(d + 1 - a) for c in range(d + 1 - a - b) if a + b + c == d])
        for exponents in exponents_combinations:
            powers = np.power(feature_values, exponents)
            new_feature = np.prod(powers, axis=1, where=(exponents != 0)).reshape(-1, 1)
            poly_features = np.concatenate((poly_features, new_feature), axis=1)
    
    return poly_features[:, 1:]


degree = int(input("Enter the Degree: "))

X_poly = create_polynomial_features_3vars(X, degree)
print("Polynomial features shape:", X_poly.shape)


# In[146]:


def Z_Score_Normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1 
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

X_normalized, mean_X, std_X = Z_Score_Normalization(X_poly)


# In[147]:


def polynomial_regression(X, y, alpha, iterations, lamda):
    num_samples, num_features = X.shape
    
    np.random.seed(42)
    weights = np.random.randn(num_features)
    b = np.random.randn(1)
    
    cost_history = []
    
    for i in range(iterations):
        y_pred = np.dot(X, weights) + b
        error = y_pred - y
        
        gradient = (1 / num_samples) * (np.dot(X.T, error) + lamda * weights)
        gradient[1:] += (lamda / num_samples) * weights[1:]  
        
   
        weights -= alpha * gradient
        b -= alpha * (np.mean(error) + lamda * np.sum(weights[1:]) / num_samples) 
        
        
        cost = (1 / (2 * num_samples)) * (np.sum(np.square(error)) + lamda * np.sum(np.square(weights)))
        cost_history.append(cost)
        
    return weights, b, cost_history


# In[148]:


alpha = 0.1
iterations = 2000
lamda = 0.001 


optimal_weights, optimal_bias, cost_history = polynomial_regression(X_normalized, y, alpha, iterations, lamda)


plt.plot(range(iterations), cost_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Number of Iterations')
plt.show()


final_cost = cost_history[-1]
print(f"Final Cost: {final_cost}")


# In[149]:


def predict_with_consistency(X, weights, bias, degree, mean_X, std_X):
    X_poly = create_polynomial_features_3vars(X, degree)  
    X_scaled = (X_poly - mean_X) / std_X
    return np.dot(X_scaled, weights) + bias

y_pred = predict_with_consistency(X, optimal_weights, optimal_bias, degree, mean_X, std_X)


# In[150]:


def r2_score(y_true, y_pred):
    mean_observed = np.mean(y_true)
    total_sum_squares = np.sum((y_true - mean_observed) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)
    return r2

r2 = r2_score(y, y_pred)
print(f"R-2 score: {r2}")
print(optimal_weights)
print(optimal_bias)


# In[151]:


plt.plot(X[:,1], y)
plt.xlabel('FEATURE 3')
plt.ylabel('TARGET VALUE')
plt.title('feature 3 vs target values')
plt.show()


# In[152]:


test_file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Polynomialdata_test.csv'
test_data = pd.read_csv(test_file_path)


X_test = test_data[['feature 1', 'feature 2', 'feature 3']].values


X_test_poly = create_polynomial_features_3vars(X_test, degree)

y_pred_test = predict_with_consistency(X_test, optimal_weights, optimal_bias, degree, mean_X, std_X)

test_data['predicted_target'] = y_pred_test

result_file_path = r'C:\Users\Abhinav\OneDrive\Desktop\data sets for woc\Predictions.csv'
test_data.to_csv(result_file_path, index=False)


# In[153]:


np.set_printoptions(threshold=np.inf)
print(y_pred_test)

