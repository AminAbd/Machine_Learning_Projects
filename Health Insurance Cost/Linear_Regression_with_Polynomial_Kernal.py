import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pylab as plt
################################################
Data=pd.read_csv('Health Insurance Cost/insurance.csv')
print(Data.head())
# Get a concise summary of the Data
print(Data.info())
# Check for missing values
print(Data.isnull().sum())

# For binary categorical variables, replace categories with 0 and 1
Data['sex'] = Data['sex'].map({'female': 0, 'male': 1})
Data['smoker'] = Data['smoker'].map({'no': 0, 'yes': 1})

# For the 'region' variable, use get_dummies to perform one-hot encoding
Data = pd.get_dummies(Data, columns=['region'], drop_first=True)
#print(Data.head())

# Define the features and the target variable
X = Data.drop('charges', axis=1)  # Replace 'charges' with the name of your target variable
Y = Data['charges']  # Replace 'charges' with the name of your target variable

##########################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Apply a second-order polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale the polynomial features
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Create and train the linear regression model on the scaled, transformed features
model_poly_scaled = LinearRegression()
model_poly_scaled.fit(X_train_poly_scaled, Y_train)

# Predict and Evaluate using the scaled, transformed features
y_pred_poly_scaled = model_poly_scaled.predict(X_test_poly_scaled)
mse_poly_scaled = mean_squared_error(Y_test, y_pred_poly_scaled)
print('MSE with scaled polynomial features:', mse_poly_scaled)


#############################
plt.figure()
plt.scatter(Y_test,y_pred_poly_scaled)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()

