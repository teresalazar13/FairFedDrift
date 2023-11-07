import tensorflow as tf


class NN_model:
    def __init__(self, dataset, seed):

        if dataset.is_large:
            self.batch_size = 64
            self.n_epochs = 10  # TODO - remove

            """
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                                  input_shape=dataset.input_shape))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
            self.model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))"""
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                             input_shape=dataset.input_shape))
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        else:
            self.batch_size = 32
            self.n_epochs = 5

            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=dataset.input_shape)
            )
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))

        if dataset.is_binary_target:
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
        else:
            self.model.add(tf.keras.layers.Dense(10, activation='softmax'))  # TODO - number of classes here

        """
        initializer = tf.keras.initializers.RandomNormal(seed=seed)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer),
        ])"""

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def compile(self, dataset):
        if dataset.is_large:
            #optimizer = tf.keras.optimizers.legacy.Adam()
            #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        if dataset.is_binary_target:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'

        self.model.compile(loss=loss, optimizer=optimizer)

    def learn(self, x, y):
        tf.keras.utils.disable_interactive_logging()
        self.model.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x)
