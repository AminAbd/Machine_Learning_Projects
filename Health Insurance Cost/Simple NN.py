import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

Data=pd.read_csv('Health Insurance Cost/insurance.csv')
#print(Data.head())
# Get a concise summary of the Data
#print(Data.info())
# Check for missing values
#print(Data.isnull().sum())

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
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, Y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)

# Predict the charges
y_pred_nn = model.predict(X_test_scaled)

# Calculate the mean squared error
mse_nn = mean_squared_error(Y_test, y_pred_nn)
print('Mean Squared Error (Neural Network):', mse_nn)

# Plotting the results
plt.figure()
plt.scatter(Y_test, y_pred_nn, label='Neural Network Predictions')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (Neural Network)')
plt.legend()
plt.show()
