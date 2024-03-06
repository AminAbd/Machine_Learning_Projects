import os

# Replace this with the actual base directory path on your system
base_directory_path = 'Sentiment analysis/aclImdb_v1/aclImdb/train'

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
    category = 'non-identified' if sub_dir == 'unsup' else sub_dir
    read_texts(os.path.join(base_directory_path, sub_dir), category)


print(texts)                
"""
# Output the number of documents read for each category
for category, text_list in texts.items():
    print(f"{category} texts: {len(text_list)}")

# At this point, you have three lists: texts['positive'], texts['negative'], texts['non-identified']
# You can proceed with further processing like tokenization or Word2Vec training.
"""