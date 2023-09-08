import tensorflow as tf


class NN_model:
    def __init__(self, input_shape, seed, is_image):
        self.batch_size = 32
        self.n_epochs = 50
        initializer = tf.keras.initializers.RandomNormal(seed=seed)
        if not is_image:
            """
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_shape,)),
                tf.keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer),
            ])"""
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                                  input_shape=input_shape))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))

        else:
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
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
            """
            self.model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
            )"""
            self.model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.legacy.Adam(),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
            )

        else:
            opt = tf.keras.optimizers.SGD(learning_rate=0.1)
            self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def learn(self, x, y, sample_weights=None):
        tf.keras.utils.disable_interactive_logging()
        if sample_weights is not None:
            self.model.fit(
                x=x, y=y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0, sample_weight=sample_weights
            )
        else:
            self.model.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x)
