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
