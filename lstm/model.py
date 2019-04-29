from input import data_generator, all_features, dev_generator, validation_set, seq_length
import tensorflow as tf

def get_model(batch_size, n_features):
    inputs = tf.keras.layers.Input(shape=(None, n_features), batch_size=batch_size)
    x = tf.keras.layers.LSTM(units=256, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(units=128)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    output = tf.keras.layers.Dense(units=4, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy(), metrics=[tf.metrics.CategoricalAccuracy()])

    return model

model = get_model(1, len(all_features()) - 2)

model.fit_generator(
    data_generator(),
    epochs=1,
    callbacks=[tf.keras.callbacks.TensorBoard(update_freq='batch')],
    steps_per_epoch=2000
)

tf.keras.models.save_model(model, 'model.hdf5')