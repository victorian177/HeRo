import json
import os
import pickle

import nltk
import numpy as np
from gensim import corpora
from keras.layers import Activation, Dense
from keras.models import Sequential, model_from_json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# nltk required documents
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Stemming and Lemmatization are the preprocessing before vectorisation
lemma = WordNetLemmatizer()
stemmer = PorterStemmer()

# stopwords contains most common words that don't provide much information e.g. 'the', 'a'
stop_words = stopwords.words("english")


def process_tokens(tokens):
    # case lowering and punctuation from word tokens
    low_words = [t.lower()
                 for t in tokens if t not in [',', '!', '.', '?']]
    # stopword removal
    words = [w for w in low_words if w not in stop_words]
    # lemmatisation
    lem_words = [lemma.lemmatize(w) for w in words]
    # stemming
    stem_words = [lemma.lemmatize(w) for w in words]

    return list(set(lem_words + stem_words))


# bag of words generator from corpus
def bag_of_words(doc_tokens, dctnry):
    bow = [0 for _ in range(len(dctnry))]
    for w in doc_tokens:
        if w in list(dctnry.keys()):
            bow[dctnry[w]] += 1

    return np.array(bow)


# If the intents file has been modified, retrain model
with open('data.pickle', 'rb') as f:
    dictionary, labels, rec_mod_time = pickle.load(f)

cur_mod_time = str(os.path.getmtime('intents.json'))

if cur_mod_time != rec_mod_time:
    print("intents.json file has been modified. Model is being retrained.")

    # PREPROCESSING
    # intents.json contains documents for generating corpus, input vectors and labels
    with open('intents.json') as file:
        data = json.load(file)

    labels = []  # set of target variables
    docs_X_prep = []  # documents prepper
    docs_X = []  # documents containing vectors used to train neural net
    docs_y = []  # array representation of target variable for corresponding docs_X

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word_tokens = word_tokenize(pattern)
            docs_X_prep.append(word_tokens)
            docs_y.append(intent["tag"])

            if intent['tag'] not in labels:
                labels.append(intent["tag"])

    # process prepped documents
    for tokens in docs_X_prep:
        docs_X.append(process_tokens(tokens))

    # corpus containing words and unique identifiers
    dictionary = corpora.Dictionary(docs_X)

    with open('data.pickle', 'wb') as f:
        pickle.dump((dictionary, labels, cur_mod_time), f)

    # conversion of documents to vectors
    training = []
    for doc in docs_X:
        t = bag_of_words(doc, dictionary.token2id)
        training.append(t)

    output = []
    for i in range(len(docs_y)):
        empty_output = [0 for _ in range(len(labels))]
        output_row = empty_output[:]
        output_row[labels.index(docs_y[i])] = 1
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("training output.pickle", 'wb') as f:
        pickle.dump((training, output), f)

    # MODEL
    # ML Model
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=10)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=10)
    y = np.array([np.argmax(i) for i in output])
    n_scores = cross_val_score(
        model, training, y, scoring='accuracy', cv=cv, n_jobs=-1)

    print(training)
    print(y)

    model.fit(training, y)

    # Deep Learning Model
#     # model containing 3 layers
#     model = Sequential([
#         Dense(units=8, input_shape=(len(training[0]), ), activation="relu"),
#         Dense(units=8, activation="relu"),
#         Dense(units=8, activation="relu"),
#         Dense(units=len(output[0]), activation="softmax")
#     ])

#     model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

#     print(model.summary())

#     model.fit(x=training, y=output, batch_size=8, epochs=200)

#     # model saving
#     model_json = model.to_json()
#     with open("model.json", "w") as json_file:
#         json_file.write(model_json)

#     model.save_weights("model.h5")

#     model.compile()

# # If intents file is not modified, load previous model
# else:
#     print("Reloading model...")
#     with open("model.json") as f:
#         model_json = f.read()

#     model = model_from_json(model_json)
#     model.load_weights("model.h5")

#     model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

#     print(model.summary())


def chat(inpt):
    # return predictions on which of the labels a text belongs to alongside their corresponding probabilities
    bow = bag_of_words(inpt, dictionary.token2id)
    result = model.predict([bow])
    mx_index = np.argmax(result)

    tag = labels[mx_index]

    return result, tag
