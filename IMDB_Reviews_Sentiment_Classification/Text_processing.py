import os
import nltk # Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Base Directory
base_directory_path = 'IMDB_Reviews_Sentiment_Classification/aclImdb_v1/aclImdb/train'

# Sub-directories for positive, negative and unsup (non-identified) texts
sub_directories = ['pos', 'neg', 'unsup']

# Dictionaries to hold text content for each category
texts = {
    'positive': [],
    'negative': [],
    'non-identified': []
}


# Function to read files from a directory and append to the respective list
def read_texts(directory, category):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts[category].append(text)

# Read texts from each sub-directory
for sub_dir in sub_directories:
    if sub_dir == 'pos':
        category = 'positive'
    elif sub_dir == 'neg':
        category = 'negative'
    else:  # sub_dir == 'unsup'
        category = 'non-identified'
    read_texts(os.path.join(base_directory_path, sub_dir), category)
      
# Output the number of documents read for each category
for category, text_list in texts.items():
    print(f"{category} texts: {len(text_list)}")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lowercase, remove non-alphanumeric characters, and lemmatize
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]
    
    # Remove stop words
    filtered_tokens = [token for token in processed_tokens if token not in stop_words]
    
    return ' '.join(filtered_tokens)

# Apply preprocessing to each text
for category in texts:
    texts[category] = [preprocess_text(text) for text in texts[category]]
#########################
import json

# Save preprocessed texts to a JSON file for later use
with open('IMDB_Reviews_Sentiment_Classification/processed_texts.json', 'w') as outfile:
    json.dump(texts, outfile)    
print("Pre-processing is Done.")    