import json
import os
import random
import re
import numpy as np
import pandas as pd
import tensorflow as tf

file_regex = re.compile('^\d+[A-Z]{2}\.csv$')
file_names = []
seq_length = 256

validation_set = []

for f in os.listdir('../'):
    if file_regex.match(f):
        file_names.append(f)

def all_features():
    with open('../features.json') as f:
        return json.load(f)

def optimal_features():
    with open('../optimal_features.json') as f:
        return json.load(f)

def get_extrema():
    with open('../extrema.json') as f:
        return json.load(f)

def dev_generator():
    while True:
        for v in validation_set:
            yield v

def data_generator():
    current_iteration = 0
    random.shuffle(file_names)
    repetitions = 24

    extrema = get_extrema()
    max = np.transpose(np.array(extrema['max']))
    min = np.transpose(np.array(extrema['min']))

    while True:
        if current_iteration % len(file_names) == 0:
            random.shuffle(file_names)

        if current_iteration % 15 == 0:
            repetitions *= 2

        name = file_names[current_iteration % len(file_names)]
        df = pd.read_csv(f'../{name}')
        df = df.append(pd.read_csv(f'../{file_names[(current_iteration + 1) % len(file_names)]}'))
        df = df.append(pd.read_csv(f'../{file_names[(current_iteration + 2) % len(file_names)]}'))

        features = df.drop(['event', 'Unnamed: 0'], axis=1).values
        targets = df['event'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3}).values

        iterations = list(range(np.size(targets) - seq_length))
        random.shuffle(iterations)

        for start in iterations[:repetitions]:
            val = random.random()

            batch_features = features[start:start + seq_length, :]

            batch_features = (batch_features - min) / (max - min)

            batch_targets = targets[start + seq_length - 1]
            batch_targets = tf.one_hot(batch_targets, depth=4)

            if val >= 0.97:
                validation_set.append((tf.expand_dims(batch_features, 0), tf.expand_dims(batch_targets, 0)))
            else:
                yield tf.expand_dims(batch_features, 0), tf.expand_dims(batch_targets, 0)

        current_iteration += 1