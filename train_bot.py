import string

import os
import random
import json
from joblib import dump
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import corpus

import numpy as np
import tensorflow


working_dir = os.getcwd()
data_dir = os.path.join(working_dir, "data")
models_dir = os.path.join(working_dir, "models")

lemmatizer = WordNetLemmatizer()
stopwords = set(corpus.stopwords.words('english'))


def load_data(intents_file):
    intents_file_path = os.path.join(data_dir, intents_file)
    with open(intents_file_path, "r") as f:
        intents_json = json.load(f)
    return intents_json


def save_file(data, file_name):
    file_path = os.path.join(models_dir, file_name)
    dump(data, file_path)


def preprocess_doc_and_extract_words(document):
    twords = word_tokenize(document)
    words_list = []
    for word in twords:
        word = lemmatizer.lemmatize(word.lower())
        if (word not in string.punctuation) and (word not in stopwords):
            words_list.append(word)

    return words_list


def extract_data_from_intents(intents):
    all_words = set()
    all_classes = set()
    all_documents = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        all_classes.add(tag)
        for pattern in intent["patterns"]:
            words = preprocess_doc_and_extract_words(str(pattern))
            all_words.update(words)
            all_documents.append((words, tag))

    return list(all_words), list(all_classes), all_documents


def build_data_for_training(documents, words, classes, encoding="binary", shuffle=True):
    train_data = []
    if encoding == "binary":
        for doc in documents:
            data, target = doc
            row = [int(w in data) for w in words]
            target_multilabel = [0] * len(classes)
            target_multilabel[classes.index(target)] = 1
            train_data.append([row, target_multilabel])
    else:
        print(f"Encoding {encoding} not supported for training data!")

    if shuffle:
        random.shuffle(train_data)
        training = np.array(train_data)
        # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    return train_x, train_y


def build_model(num_features):
    ip = Input((num_features,), dtype=tf.int32)
    x = Dense(32, activation="relu")(ip)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Dense(8, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)


if __name__ == "__main__":
    print("Training the bot")

    intents_file = "intents.json"
    intents = load_data(intents_file)

    print("load data from intents")
    words, classes, documents = extract_data_from_intents(intents)

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    save_file(words, "words.h5")
    save_file(classes, "classes.h5")

    print("build training data")
    X, y = build_data_for_training(documents, words, classes, encoding="binary")
    print(f"Training data has {len(X)} observations, and {len(X[0])} features")

