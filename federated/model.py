import tensorflow as tf
from classification_models.keras import Classifiers


class NN_model:
    def __init__(self, dataset, seed):
        initializer = tf.keras.initializers.RandomNormal(seed=seed)

        if not dataset.is_image:  # Adult
            self.batch_size = 10
            self.n_epochs = 10
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=dataset.input_shape))
            self.model.add(tf.keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer))
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
        else:
            if dataset.is_large:  # CIFAR-100 - ResNet
                self.batch_size = 16
                self.n_epochs = 10
                ResNet18, preprocess_input = Classifiers.get('resnet18')
                base_model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
                # Add custom layers for CIFAR-100
                x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                output = tf.keras.layers.Dense(100, activation='softmax')(x)
                self.model = tf.keras.Model(inputs=base_model.input, outputs=output)

            else:  # MNIST and FEMNIST
                self.batch_size = 32
                self.n_epochs = 5
                self.model = tf.keras.models.Sequential()
                self.model.add(tf.keras.layers.Conv2D(
                    32, (3, 3), activation='relu', kernel_initializer=initializer, input_shape=dataset.input_shape)
                )
                self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                self.model.add(tf.keras.layers.Flatten())
                self.model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer))
                self.model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer))

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def compile(self, dataset):
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
