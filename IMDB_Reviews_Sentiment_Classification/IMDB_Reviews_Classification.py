import dill
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load preprocessed training texts from the dill file
with open('IMDB_Reviews_Sentiment_Classification/training_texts.dill', 'rb') as infile:
    training_texts = dill.load(infile)

# Load preprocessed testing texts from the dill file
with open('IMDB_Reviews_Sentiment_Classification/testing_texts.dill', 'rb') as infile:
    testing_texts = dill.load(infile)

# Feature Extraction: Creating a bag-of-words model using training data
vectorizer = CountVectorizer()
# Combine positive and negative categories from training data
train_corpus = training_texts['positive'] + training_texts['negative']
X_train = vectorizer.fit_transform(train_corpus)
y_train = [1] * len(training_texts['positive']) + [0] * len(training_texts['negative'])

# Transform testing data using the same vectorizer
# Combine positive and negative categories from testing data
test_corpus = testing_texts['positive'] + testing_texts['negative']
X_test = vectorizer.transform(test_corpus)
y_test = [1] * len(testing_texts['positive']) + [0] * len(testing_texts['negative'])

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
