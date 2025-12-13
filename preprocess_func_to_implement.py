# ----x----x----x----x----x----x----x
#      creat this function in flask applicaton (__main__)
# ----x----x----x----x----x----x----x
# dont import this file in flask app only copy and paste this function in app.py


import nltk
from nltk.stem import PorterStemmer   # port stemmer
import re
import pandas as pd
import string

# get stop words
with open('stopwords.txt', 'r') as f:
  stop_words = f.read()

stopwords = stop_words.split()

def preprocess_text(text):
    # convert text to lower case
    text = text.lower()

    # remove punctuation and stopwords
    patt = f'[{string.punctuation}]' + '|' + '\\b(' + f'{'|'.join(stopwords)}' + ')\\b'
    text = re.sub(patt, ' ', text)

    # tokenize
    text = nltk.word_tokenize(text)

    # stemming
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    # convert text(list) into string
    return ' '.join(text)