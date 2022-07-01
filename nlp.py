import json
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# PREPROCESSING
is_new = True
with open('vocab.json') as f:
    rec_vocab = json.load(f)

dataset_df = pd.read_csv("dataset.csv")

tfidf = TfidfVectorizer(stop_words='english')
features = tfidf.fit_transform(dataset_df['statement']).toarray()
cur_vocab = tfidf.vocabulary_

if rec_vocab == cur_vocab:
    vocab = rec_vocab
    is_new = False
else:
    with open("vocab.json", 'w') as f:
        json.dump(cur_vocab, f)
    vocab = cur_vocab


# MODEL TRAINING
if is_new:
    multi_nb = MultinomialNB()
    multi_nb.fit(features, np.array(dataset_df["response"]))

    with open('model.pickle', 'wb') as mdl:
        pickle.dump(multi_nb, mdl)

else:
    with open('model.pickle', 'rb') as mdl:
        multi_nb = pickle.load(mdl)

# MODEL PREDICTING
def predictor(inpt):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit_transform()
    multi_nb.predict()
