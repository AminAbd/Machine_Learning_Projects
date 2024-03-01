import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
#  Read the data #######
Data=pd.read_csv('Gender Classification/gender_classification_v7.csv')
Data['gender'] = Data['gender'].map({'Female': 0, 'Male': 1})
X = Data.drop('gender', axis=1)  # Replace 'charges' with the name of your target variable
Y = Data['gender']  # Replace 'charges' with the name of your target variable

###############################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# Initialize and train the XGBClassifier model
model = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, Y_train)

# Predict the gender
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities of the positive class
y_pred_labels = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on a threshold of 0.5

# Calculate the accuracy
accuracy = accuracy_score(Y_test, y_pred_labels)
print(f'Accuracy: {accuracy:.4f}')

# Confusion Matrix
cm = confusion_matrix(Y_test, y_pred_labels)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()