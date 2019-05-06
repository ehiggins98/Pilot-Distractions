import json
import tensorflow as tf
import input
from model import get_model

batch_size = 64
features = input.get_optimal_features()

model = get_model(len(features), batch_size, [32, 64, 128, 256, 128, 64, 32, 16])
model.fit(
    x=input.get_dataset('train.tfrecord', batch_size, features, input.match_extrema(features)),
    epochs=1,
    shuffle=True,
    steps_per_epoch=int(4800000/batch_size)
)

model.evaluate(
    x=input.get_dataset('dev.tfrecord', batch_size, features, input.match_extrema(features)),
    steps=int(48000/batch_size)
)

model.save_weights('model.hdf5')