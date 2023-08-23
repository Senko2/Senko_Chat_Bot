import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensor.keras.models import Sequentail
from tensor.keras.layers import Dense , Activation, Dropout
from tensor.keras.optimizers import SGD 

lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words = []
classes =[]
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in  intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intents['tags']))
        if intent['tag'] not  in classes:
            classes.append(intents['tag'])

words = [lemmatizer.lemmatize(word) for word in words  if  word not in ignore_letters]

words = sorted(set(words))
