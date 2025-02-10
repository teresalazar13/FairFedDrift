import tensorflow as tf
from classification_models.keras import Classifiers
from keras.applications.resnet50 import ResNet50



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
                self.batch_size = 64
                self.n_epochs = 30
                self.model = tf.keras.models.Sequential()
                self.model.add(tf.keras.layers.Resizing(224, 224, interpolation='bilinear'))
                resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                for layer in resnet_model.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = True
                    else:
                        layer.trainable = False
                self.model.add(resnet_model)
                self.model.add(tf.keras.layers.GlobalAveragePooling2D())
                self.model.add(tf.keras.layers.Dense(256, activation='relu'))
                self.model.add(tf.keras.layers.Dropout(.25))
                self.model.add(tf.keras.layers.BatchNormalization())
                self.model.add(tf.keras.layers.Dense(100, activation='softmax'))
                dummy_input = tf.random.normal([1] + list(dataset.input_shape))
                dummy_labels = tf.zeros([1, 100])
                self.compile(dataset)
                self.model.fit(dummy_input, dummy_labels, epochs=1, batch_size=1, verbose=0)

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
        if dataset.is_large:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
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
