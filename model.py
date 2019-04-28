import tensorflow as tf

def get_model(n_features, batch_size):
    inputs = tf.keras.layers.Input(shape=(n_features,), batch_size=batch_size)
    x = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)(inputs)
    x = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)(inputs)
    x = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(8, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy())
    return model