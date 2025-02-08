import tensorflow as tf


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
                self.batch_size = 32
                self.n_epochs = 5
                self.model = ResNet18(dataset.input_shape, 100)

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


# Conv1 (7×7, stride 2)
# MaxPooling (3×3, stride 2)
# 4 Residual Block Groups with increasing filters: [64, 128, 256, 512]
# Global Average Pooling to reduce feature maps
def ResNet18(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs, outputs)

    return model


# Each block has two 3×3 Conv layers with batch norm and ReLU
# Skip connection (Add()) to merge input with output
# Uses Conv2D(1x1, stride=2) for downsampling when needed
def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    y = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if downsample:
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.layers.ReLU()(out)

    return out
