import tensorflow as tf
import numpy as np


class NN_model_online:
    def __init__(self, n_features, seed, is_image):
        initializer = tf.keras.initializers.RandomNormal(seed=seed)
        if not is_image:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(n_features,)),
                tf.keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer),
            ])
        else:
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.Dense(10, activation='softmax'))

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def compile(self, is_image):
        if not is_image:
            self.model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.0001),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
            )
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
            self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def learn_one(self, x_i, y_i):
        data = tf.data.Dataset.from_tensor_slices((list([x_i]), list([y_i]))).batch(1)
        tf.keras.utils.disable_interactive_logging()
        self.model.fit(data, epochs=1, verbose=0)

    def predict_one(self, x_i):
        return self.model.predict(np.array([x_i]))
