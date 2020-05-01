import os
import random
import json
import nltk
nltk.download('punkt')
nltk.download('wordnet')


working_dir = os.getcwd()
data_dir = os.path.join(working_dir, "data")


def load_data(intents_file):
    intents_file_path = os.path.join(data_dir, intents_file)
    with open(intents_file_path, "r") as f:
        intents_json = json.load(f)
    return intents_json


if __name__ == "main":
    print("Training the bot")

    intents_file = "intents.json"
    intents_json = load_data(intents_file)
    print(intents_json)
