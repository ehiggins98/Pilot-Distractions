import numpy as np
import tensorflow as tf
import json
import input
from model import get_model

def feature_importances(model):
    importances = []
    input_weights = model.layers[1].get_weights()[0]
    print(np.shape(input_weights))
    for i in range(np.shape(input_weights)[0]):
        importances.append(np.sum(np.absolute(input_weights[i, :])))

    return importances

batch_size = 32

model = get_model(25, batch_size)
model.fit(
    x=input.get_dataset('train.tfrecord', batch_size, input.get_features(), input.get_extrema()),
    epochs=1,
    shuffle=True,
    steps_per_epoch=9000)

loss = model.evaluate(
    x=input.get_dataset('dev.tfrecord', batch_size, input.get_features(), input.get_extrema()),
    steps=int(48000/batch_size)
)

importances = [(importance, i) for i, importance in enumerate(feature_importances(model))]
importances.sort(key=lambda x: x[0], reverse=True)

min_loss = loss
argmin = input.get_features()

for n_features in range(1, 25):
    subset = importances[:n_features]
    subset = list(map(lambda x: x[1], subset))
    features = []
    
    for i, f in enumerate(input.get_features()):
        if i in subset:
            features.append(f)

    print(f'Testing features: {features}')

    extrema = input.get_extrema()
    extrema = {'min': np.array(extrema['min'])[subset], 'max': np.array(extrema['max'])[subset]}

    avg_loss = 0

    for i in range(3):
        model = get_model(n_features, batch_size)
        model.fit(
            x=input.get_dataset('train.tfrecord', batch_size, features, extrema),
            epochs=1,
            shuffle=True,
            steps_per_epoch=9000)
        
        loss = model.evaluate(
            x=input.get_dataset('dev.tfrecord', batch_size, features, extrema),
            steps=int(48000/batch_size)
        )

        avg_loss += loss / 3

    if avg_loss < min_loss:
        argmin = features
        min_loss = loss

print(f'Best features: {argmin}')

with open('optimal_features.json', 'w') as f:
    json.dump(argmin, f)