import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
# Read the Houding Dataset ######
Data=pd.read_csv('Housing/Housing_Data.csv')
# Print the data ###
#print(Data)
#print(Data.keys())
# Split the dataset into features (X) and the target variable (y)
X = Data.drop('House Price ($)', axis=1)
y = Data['House Price ($)']
###############
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


############


# Assuming y_test and y_pred are already defined
# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual House Prices ($)')
plt.ylabel('Predicted House Prices ($)')
plt.title('Actual vs Predicted House Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()
