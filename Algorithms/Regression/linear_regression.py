# importing the libraries
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

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\nTraining Set - Matrix of Feature Vector, X_train: \n", X_train)
print("\nTraining Set - Dependent Variable Vector, y_train: \n", y_train)
print("\nTesting Set - Matrix of Feature Vector, X_test: \n", X_test)
print("\nTesting Set - Dependent Variable Vector, y_test: \n", y_test)

# training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)
print("\nTest Set Results: \n", y_pred)

# visualising the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary VS Experience ( Training Set )")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualising the test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary VS Experience ( Test Set )")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
#  Notice that the value of the feature (12 years) was input in a double pair of square brackets.
#  That's because the "predict" method always expects a 2D array as the format of its inputs.
#  And putting 12 into a double pair of square brackets makes the input exactly a 2D array
print("\nthe salary of an employee with 12 years of experience: ", regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients.
# Linear Equation -> y = b0 + b1 * x1
# Slope ( b1 ), alias 'coefficient' ( coef_ )
# y intercept ( b0 ), alias 'intercept' ( intercept_ )
print("Coefficient or coef ( alias slope ): ", regressor.coef_)
print("intercept or y_intercept: ", regressor.intercept_)

# Therefore, the equation of our simple linear regression model is:
# Salary=9345.94×YearsExperience+26816.19
# To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object.
# Attributes in Python are different from methods and usually return a simple value or an array of values.
