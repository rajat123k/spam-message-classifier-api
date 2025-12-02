from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import PorterStemmer   # port stemmer
import re
import nltk
import pandas as pd


class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, punctuation, stopwords):
      self.column = column
      self.punctuation = punctuation
      self.stemmer = PorterStemmer()
      self.stopwords = stopwords
      patt = f'[{self.punctuation}]' + '|' + '\\b(' + f'{'|'.join(self.stopwords)}' + ')\\b'
      self.stopw_punc_patt = re.compile(patt)

    def fit(self, X=None, y=None):
        # no learning from X
        return self

    def transform(self, X):
      '''transforming text Dataframes columns(string)'''
      column = self.column

      final_text = pd.Series(name=column)
      assert isinstance(X, pd.DataFrame), 'passed variabale is not Dataframe'

      for i in range(X.shape[0]):
        try:
          # Access elements by integer position using .iloc to handle non-contiguous indices
          text = X[column].iloc[i]
          # convert text to lower case
          text = text.lower()

          # remove punctuation and stopwords
          text = self.stopw_punc_patt.sub(' ', text)

          # tokenize
          text = nltk.word_tokenize(text)

          # stemming
          text = [self.stemmer.stem(word) for word in text]

          # convert text(list) into string
          text = ' '.join(text)

          # store transformed text
          final_text[len(final_text)] = text
        except:
          print(text)
          return 0
      return final_text