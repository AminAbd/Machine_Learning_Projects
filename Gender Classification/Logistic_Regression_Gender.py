import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#  Read the data #######
Data=pd.read_csv('Gender Classification/gender_classification_v7.csv')
Data['gender'] = Data['gender'].map({'Female': 0, 'Male': 1})
X = Data.drop('gender', axis=1)  # Replace 'charges' with the name of your target variable
Y = Data['gender']  # Replace 'charges' with the name of your target variable

###############################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# Initialize the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
# Fit the model to the training data
logreg.fit(X_train, Y_train)
# Predict on the test data
Y_pred = logreg.predict(X_test)
# Calculate the accuracy
accuracy = accuracy_score(Y_test, Y_pred)

# Print the accuracy and the classification report
print(f'Accuracy: {accuracy}')
# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()