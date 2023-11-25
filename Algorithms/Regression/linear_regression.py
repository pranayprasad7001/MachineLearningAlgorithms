# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('../../Datasets/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("\nMatrix of Feature Vector, X: \n", X)
print("\nDependent Variable Vector, Y: \n", y)

# Splitting the dataset into the training set and testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("\nTraining Set - Matrix of Feature Vector, X_train: \n", X_train)
print("\nTraining Set - Dependent Variable Vector, y_train: \n", y_train)
print("\nTesting Set - Matrix of Feature Vector, X_test: \n", X_test)
print("\nTesting Set - Dependent Variable Vector, y_test: \n", y_test)