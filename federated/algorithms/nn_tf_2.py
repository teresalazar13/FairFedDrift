import tensorflow as tf
from tensorflow.keras.applications import ResNet50

class NNTF_2:
    def __init__(self, dataset, seed):
        initializer = tf.keras.initializers.RandomNormal(seed=seed)

        self.batch_size = 64
        self.n_epochs = 15
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Resizing(224, 224, interpolation='bilinear'))
        resnet_model50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in resnet_model50.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        self.model.add(resnet_model50)
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(.25))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(100, activation='softmax'))
        dummy_input = tf.random.normal([1] + list(dataset.input_shape))
        dummy_labels = tf.zeros([1, 100])
        self.compile(dataset)
        self.model.fit(dummy_input, dummy_labels, epochs=1, batch_size=1, verbose=0)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def compile(self, _):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=optimizer)

    def learn(self, x, y):
        tf.keras.utils.disable_interactive_logging()
        self.model.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x)
