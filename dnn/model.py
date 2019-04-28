import tensorflow as tf

def get_model(n_features, batch_size, layer_sizes):
    inputs = tf.keras.layers.Input(shape=(n_features,), batch_size=batch_size)
    x = inputs
    for s in layer_sizes:
        x = tf.keras.layers.Dense(s, activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy(), metrics=[tf.metrics.CategoricalAccuracy()])
    return model