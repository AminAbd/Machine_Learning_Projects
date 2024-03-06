import os
import dill
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]
    filtered_tokens = [token for token in processed_tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

# Function to read and preprocess texts from a directory
def read_and_preprocess_texts(base_directory_path, sub_directories):
    texts = {}
    for sub_dir in sub_directories:
        category = 'positive' if sub_dir == 'pos' else 'negative' if sub_dir == 'neg' else 'non-identified'
        directory_path = os.path.join(base_directory_path, sub_dir)
        texts[category] = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    processed_text = preprocess_text(text)
                    texts[category].append(processed_text)
    return texts

# Preprocess and save training data
train_base_directory = 'IMDB_Reviews_Sentiment_Classification/aclImdb_v1/aclImdb/train'
train_sub_directories = ['pos', 'neg', 'unsup']
training_texts = read_and_preprocess_texts(train_base_directory, train_sub_directories)

with open('IMDB_Reviews_Sentiment_Classification/training_texts.dill', 'wb') as outfile:
    dill.dump(training_texts, outfile)

# Preprocess and save testing data
test_base_directory = 'IMDB_Reviews_Sentiment_Classification/aclImdb_v1/aclImdb/test'
test_sub_directories = ['pos', 'neg']
testing_texts = read_and_preprocess_texts(test_base_directory, test_sub_directories)

with open('IMDB_Reviews_Sentiment_Classification/testing_texts.dill', 'wb') as outfile:
    dill.dump(testing_texts, outfile)

print("Preprocessing and saving of training and testing data is done.")
