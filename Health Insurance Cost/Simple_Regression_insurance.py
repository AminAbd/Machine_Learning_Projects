import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pylab as plt
################################################
Data=pd.read_csv('Insurance/insurance.csv')
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
# Create and train the linear regression model
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error( Y_test,y_pred)
print('mse',mse)

#############################
plt.figure()
plt.scatter(Y_test,y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()

