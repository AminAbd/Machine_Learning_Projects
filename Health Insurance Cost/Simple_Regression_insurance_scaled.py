import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

################################################
data=pd.read_csv('Linear_Regression/Insurance/insurance.csv')
# For binary categorical variables, replace categories with 0 and 1
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
# For the 'region' variable, use get_dummies to perform one-hot encoding
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Define the features and the target variable
X = data.drop('charges', axis=1)  # Replace 'charges' with the name of your target variable
Y = data['charges']  # Replace 'charges' with the name of your target variable
##########################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler to the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)
######################
model=LinearRegression()
model.fit(X_train_scaled,Y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error( Y_test,y_pred)
print('mse',mse)


import matplotlib.pylab as plt
plt.figure()
plt.scatter(Y_test,y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()




"""

# Create a linear regression model
model = LinearRegression()


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the pipeline to the training data
model.fit(X_train, y_train)

# Predict the charges on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)

# Output the Mean Squared Error and the R^2 value
print(f"Mean Squared Error: {mse}")

############
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are already defined
# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual House Prices ($)')
plt.ylabel('Predicted House Prices ($)')
plt.title('Actual vs Predicted House Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()
"""