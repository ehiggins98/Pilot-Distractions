import json
import numpy as np
import tensorflow as tf

def get_features():
    with open('features.json') as f:
        features = json.load(f)
        features.remove('crew')
        return features

def get_extrema():
    with open('extrema.json') as f:
        return json.load(f)

def match_extrema(features):
    extrema = {'min': [], 'max': []}
    all_extrema = get_extrema()

    for i, f in enumerate(get_features()):
        if f in features:
            extrema['min'].append(all_extrema['min'][i])
            extrema['max'].append(all_extrema['max'][i])

    return extrema

def get_optimal_features():
    with open('optimal_features.json') as f:
        return json.load(f)

def get_dataset(filename, batch_size, input_features, extrema):
    def parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, feature_dict)

    def to_tuple(data):
        features = [val for key, val in data.items() if key in input_features and key not in ['crew', 'event']]
        return (tf.stack(features), tf.one_hot(tf.cast(data['event'], tf.uint8), depth=4))

    def normalize(features, label):
        return ((features - extrema['min']) / (np.array(extrema['min']) - np.array(extrema['max'])), label)

    raw_data = tf.data.TFRecordDataset(filename)

    feature_dict = {x: tf.io.FixedLenFeature([], tf.float32) for x in get_features()}

    dataset = raw_data.map(parse_example)
    dataset = dataset.map(to_tuple)
    dataset = dataset.map(normalize)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size*8)
    dataset = dataset.batch(batch_size)

    return dataset