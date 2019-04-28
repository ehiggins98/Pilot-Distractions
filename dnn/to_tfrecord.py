import numpy as np
import pandas as pd
import tensorflow as tf
import json

train = pd.read_csv('train.csv')

print('Loaded CSVs')

train = train.drop('experiment', axis=1)

with open('features.json', 'w') as f:
    json.dump(train.columns.tolist(), f)

train['event'] = train['event'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
train = train.sample(len(train.index)).reset_index(drop=True)

print('Shuffled training data')

dev = train.loc[:int(0.01*len(train.index))]
train = train.loc[int(0.01*len(train.index)):]

def write_data(data, file):
    with tf.io.TFRecordWriter(file) as writer:
        for i, row in data.iterrows():
            if i % 100000 == 0:
                print(f'Writing example {i} for {file}')

            features = {}
            for col in row.index:
                features[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[col]]))
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

write_data(train, 'train.tfrecord')
write_data(dev, 'dev.tfrecord')