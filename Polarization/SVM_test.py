# -*- coding: UTF-8 -*-

from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import sklearn.metrics
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk
import pickle

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('dataset.csv',
    header=0,
    names=['DC', 'title', 'body'],
    #names=['text_raw', 'SE', 'AC', 'DC', 'DE', 'AE'],
    encoding='UTF-8')

df['text_raw'] = df[['title', 'body']].apply(lambda x: ' '.join(x), axis=1)

stemmer = PorterStemmer()

def stem_sentences(sentence):
    tokens = str(sentence).split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['text'] = df['text_raw'].apply(stem_sentences)


for i in range (len(df.DC)):
    texts.append(df['text'][i])
    labels.append(df['DC'][i])

'''
for i in range (len(testdf.DC)):
    test_texts.append(testdf['text'][i])
    test_labels.append(testdf['DC'][i])
'''

t1 = datetime.now()
vectorizer = TfidfVectorizer(analyzer= 'word', encoding='UTF-8', strip_accents='unicode', ngram_range= (1, 7), min_df=3)
classifier = LinearSVC()
Xs = vectorizer.fit_transform(texts)



print(datetime.now() - t1)
print(Xs.shape)

kappa =  sklearn.metrics.make_scorer(sklearn.metrics.cohen_kappa_score)

score = cross_val_score(classifier, Xs, labels, scoring='f1', cv=10, n_jobs=1)

print(score)
print(np.average(score))
'''
model = classifier.fit(Xs, labels)

predictions = model.predict(test_Xs)

print(np.mean(predictions == test_labels))
'''


'''
print(datetime.now() - t1)
print(score)
print(sum(score) / len(score))
'''