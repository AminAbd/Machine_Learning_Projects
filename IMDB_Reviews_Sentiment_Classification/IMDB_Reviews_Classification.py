import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load preprocessed texts from the JSON file
with open('IMDB_Reviews_Sentiment_Classification/processed_texts.json', 'r') as infile:
    texts = json.load(infile)

# Feature Extraction: Creating a bag-of-words model
vectorizer = CountVectorizer()
corpus = texts['positive'] + texts['negative']  # Combine positive and negative categories
X = vectorizer.fit_transform(corpus)

# Labeling: Assuming binary classification with 'positive' = 1 and 'negative' = 0
y = [1] * len(texts['positive']) + [0] * len(texts['negative'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{cm}")
