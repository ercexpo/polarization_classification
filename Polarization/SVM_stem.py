# -*- coding: UTF-8 -*-

from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import sklearn.metrics
import nltk
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


nltk.download()

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('train.csv',
    header=0,
    # names=['label', 'text'],
    names=['text', 'SE', 'AC', 'DC', 'DE', 'AE'],
    encoding='UTF-8')

testdf = df.tail(60)
df = df.head(len(df.text)-60)

for i in range (len(df.DC)):
    texts.append(df['text'][i])
    labels.append(df['DC'][i])

for i in range (len(testdf.DC)):
    test_texts.append(testdf['text'][i])
    test_labels.append(testdf['DC'][i])
'''
stemmer = nltk.stem.RSLPStemmer()
texts_stemmed = stemmer.stem(texts)
'''

'''
train_y, train_x = load_data_frame('train.csv')

test_y, test_x = load_data_frame('val.csv')
'''

# Stemming Code

import nltk


stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])


text_mnb_stemmed = text_mnb_stemmed.fit(texts, labels)

predicted_mnb_stemmed = text_mnb_stemmed.predict(test_texts)

np.mean(predicted_mnb_stemmed == test_labels)



'''
print(datetime.now() - t1)
print(score)
print(sum(score) / len(score))
'''