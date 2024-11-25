import tensorflow as tf


class InstanceAssignmentModel:
    def __init__(self, dataset, seed):
        initializer = tf.keras.initializers.RandomNormal(seed=seed)
        self.batch_size = 10
        self.n_epochs = 10
        self.initializer = initializer
        self.input_shape = (794,)  # TODO
        self.num_classes = 2  # Start with one class + undecided class
        self._initialize_model()

    def _initialize_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        self.model.add(tf.keras.layers.Dense(4, activation='tanh', kernel_initializer=self.initializer))
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer=self.initializer))

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def compile(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=optimizer)

    def learn(self, x, y):
        tf.keras.utils.disable_interactive_logging()
        self.model.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x)

    def add_class(self):
        self.num_classes += 1
        new_output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        self.model.pop()  # Remove the last layer (output layer)
        self.model.add(new_output_layer)  # Add the new output layer

        old_weights = self.model.get_weights()
        new_weights = self.model.get_weights()
        for i in range(len(new_weights) - 2):  # Exclude the output layer weights and biases
            new_weights[i] = old_weights[i]
        new_weights[-2][:, :-1] = old_weights[-2][:, :] # Copy weights for the existing classes
        new_weights[-2][:, -1] = tf.random.normal(shape=(old_weights[-2].shape[0],))  # Randomly initialize the new weights for the new class (last column in the weight matrix)
        new_weights[-1][:-1] = old_weights[-1]  # Preserve old biases
        new_weights[-1][-1] = 0  # Set the new bias to zero for the new class
        self.model.set_weights(new_weights)
