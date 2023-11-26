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
'''We are using SimpleImputer to handle the missing values. In Attribute 'Missing_values' we need to mention what 
kind of value we want to handle, for handling the NaN value we can use either 'numpy.nan' or 'pandas.NA'. In 
Attribute 'strategy' we can mention what kind of strategy we want to implement, for example we can mention mean, 
median, most_frequent, etc. some strategy can only work only on numeric data or str data, or both. mean & median 
works on numeric data.'''

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("\nMatrix of Feature Vector after imputing, X: \n", X)

# Encoding the categorical data

# Encoding the independent variable
'''For encoding the column in the independent variable containing the categorical data, we use the ColumnTransformers 
& OneHotEncoder. we are using the OneHotEncoder because if we use LabelEncoder which uses a order style 0,1,
2... sequence, it may hamper our machine learning model. It may indicate a order and priority which may be biased, 
so we are using OneHotEncoder to transform the categorical value in 001, 101, like style. while column Transformers 
convert one column to three column to accommodate the value for example 001 separately as 0, 0, 1 each value in three 
different columns. Also ColumnTransformers doesn't return in array value so we need to type cast that as well.'''

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("\nMatrix of Feature Vector after encoding, X: \n", X)

# Encoding the dependent variable
'''We are using the label encoder to label the dependent variable vector containing categorical data yes or no to 0 
or 1 (binary)'''

le = LabelEncoder()
y = le.fit_transform(y)

print("\nDependent Variable Vector after encoding, y: \n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nTraining Set - Matrix of Feature Vector, X_train: \n", X_train)
print("\nTraining Set - Dependent Variable Vector, y_train: \n", y_train)
print("\nTesting Set - Matrix of Feature Vector, X_test: \n", X_test)
print("\nTesting Set - Dependent Variable Vector, y_test: \n", y_test)

# Feature Scaling
'''We are scaling the features to bring them on the same case as difference between units impact the learning model, 
and we generally scale the features after the split as to avoid the data leakage of test set. While on scaling 
training set is used inside fit_transform to scale the training set but only transform is used on the test data as to 
scale the test data on the same scale of the training set not test set having its own scaling fn of its own.'''

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print("\nX_train after feature scaling: \n", X_train)
print("\nX_test after feature scaling: \n", X_test)
