import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

with open("intents.json") as json_obj:
    dataset_file = json.load(json_obj)

dataset_dict = {'statement': [], "response": []}

for record in dataset_file['intents']:
    patterns = record['patterns']
    for pattern in patterns:
        dataset_dict['statement'].append(pattern)
        dataset_dict['response'].append(record["tag"])

dataset_df = pd.DataFrame(dataset_dict)

tfidf = TfidfVectorizer(stop_words='english')
features = tfidf.fit_transform(dataset_df['statement']).toarray()
vocab = tfidf.vocabulary_

# with open("vocab.json", 'w') as f:
#     json.dump(vocab, f)

print(vocab)

multi_nb = MultinomialNB()
multi_nb.fit(features, np.array(dataset_df["response"]))

with open('vocab.json') as f:
    vcb = json.load(f)

tfidf = TfidfVectorizer(stop_words='english', vocabulary=vcb)
vec = tfidf.fit_transform(['i feel kinda sickly'])

print(multi_nb.predict(vec))
