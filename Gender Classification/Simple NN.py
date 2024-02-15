import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#  Read the data #######
Data=pd.read_csv('Gender Classification/gender_classification_v7.csv')
Data['gender'] = Data['gender'].map({'Female': 0, 'Male': 1})
X = Data.drop('gender', axis=1)  # Replace 'charges' with the name of your target variable
Y = Data['gender']  # Replace 'charges' with the name of your target variable

###############################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
# The loss function for binary classification
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(X_train_scaled, Y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Predict the charges
Y_pred_nn = model.predict(X_test_scaled)
Y_pred_nn_labels = (Y_pred_nn > 0.5).astype(int)
# Calculate the accuracy
accuracy = accuracy_score(Y_test, Y_pred_nn_labels)

# Print the accuracy and the classification report
print(f'Accuracy: {accuracy}')
# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred_nn_labels)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()