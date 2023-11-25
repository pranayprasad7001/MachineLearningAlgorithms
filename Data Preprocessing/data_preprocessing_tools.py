# importing the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv('../Datasets/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("\nMatrix of Feature Vector, X: \n", X)
print("\nDependent Variable Vector, y: \n", y)

# Handling the Missing Values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("\nMatrix of Feature Vector after imputing, X: \n", X)

# Encoding the categorical data

# Encoding the independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("\nMatrix of Feature Vector after encoding, X: \n", X)

# Encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

print("\nDependent Variable Vector after encoding, y: \n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nTraining Set - Matrix of Feature Vector, X_train: \n", X_train)
print("\nTraining Set - Dependent Variable Vector, y_train: \n", y_train)
print("\nTesting Set - Matrix of Feature Vector, X_test: \n", X_test)
print("\nTesting Set - Dependent Variable Vector, y_test: \n", y_test)

# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print("\nX_train after feature scaling: \n", X_train)
print("\nX_test after feature scaling: \n", X_test)
