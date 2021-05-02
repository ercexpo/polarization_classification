# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('train_augment_data.csv',
                 header=0,
                 nrows=50000,
                 encoding='UTF-8')

testdf = pd.read_csv('test_augment_data.csv',
                     header=0,
                     nrows=50000,
                     encoding='UTF-8')

for i in range(len(df.DC)):
    texts.append(df['text_raw'][i])
    labels.append(df['DC'][i])

for i in range(len(testdf.DC)):
    test_texts.append(testdf['text_raw'][i])
    test_labels.append(testdf['DC'][i])

classifier = Pipeline([('vect', TfidfVectorizer(analyzer='char',
                                                encoding='UTF-8',
                                                strip_accents='unicode',
                                                ngram_range=(1, 7),
                                                min_df=3)),
                       ('clf', LinearSVC(C=1.0))])

classifier = classifier.fit(df.text_raw, df.DC)

filename = 'SVM_politics_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))

predicted_svm = classifier.predict(testdf.text_raw)

print(np.mean(predicted_svm == testdf.DC))

np.savetxt('predictions.csv', predicted_svm, fmt='%d', delimiter=',')
