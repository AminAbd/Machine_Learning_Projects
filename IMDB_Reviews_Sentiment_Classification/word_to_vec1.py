from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Sample sentences
sentences = [
    "Gensim is a Python library for topic modelling, document indexing, and similarity retrieval with large corpora.",
    "Word2Vec is a method to construct such an embedding.",
    "It can be obtained using two methods, either continuous bag of words (CBOW) or skip gram model.",
    "Gensim word vectors are of length 100 by default but can be set via the size parameter.",
    "Word2Vec model can be trained with hierarchical softmax and/or negative sampling."
]

# Tokenizing the sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Training the Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Accessing the vector for a word
word_vector = model.wv['gensim']

# Finding most similar words
similar_words = model.wv.most_similar('gensim', topn=5)

print("Vector for the word 'gensim':\n", word_vector)
print("\nMost similar words to 'gensim':\n", similar_words)
