import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Health Insurance Cost/insurance.csv')

# For binary categorical variables, replace categories with 0 and 1
data['sex'] = data['sex'].map({'female': 0, 'male': 1})

# Use get_dummies to perform one-hot encoding
data = pd.get_dummies(data, columns=['smoker', 'region'], drop_first=True)

# Define the features and the target variable
X = data.drop('charges', axis=1)
Y = data['charges']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Initialize and train the XGBRegressor model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, Y_train)

# Predict the charges
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(Y_test, y_pred)
print('Mean Squared Error:', mse)

# Plotting the results
plt.figure()
plt.scatter(Y_test, y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()
