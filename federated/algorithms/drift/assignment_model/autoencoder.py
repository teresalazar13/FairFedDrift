import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam


class Autoencoder(Model):
    def __init__(self):
        image_shape = (28, 28, 1)  # Grayscale images of size 28x28  # TODO
        super(Autoencoder, self).__init__()
        self.optimizer = Adam()

        # Encoder for the image input (X)
        self.encoder_image = tf.keras.Sequential([
            Input(shape=image_shape, name="image_input"),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
        ])

        # Decoder to reconstruct the image from the joint representation
        self.decoder = tf.keras.Sequential([
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            # Output layer should produce a flattened vector for each image (batch_size * 28 * 28)
            Dense(28 * 28, activation="sigmoid"),  # 28*28 pixels per image, for each example in the batch  # TODO
            Reshape((28, 28))  # Reshape the flattened vector back to (28, 28) for each image  # TODO
        ])

    def call(self, inputs):
        x, y = inputs  # x is image input, y is one-hot encoded labels
        x_encoded = self.encoder_image(x)
        joint_representation = Concatenate()([x_encoded, y])  # Concatenate image and label representations
        reconstruction = self.decoder(joint_representation)

        return reconstruction

    def train_on_batch(self, x_batch, y_batch):
        """
        Train the autoencoder on a single batch and return joint embeddings.
        """
        with tf.GradientTape() as tape:
            reconstruction = self.call([x_batch, y_batch])
            loss = tf.reduce_mean(tf.square(x_batch - reconstruction))  # Reconstruction loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return the joint embeddings after training (for concept assignment)
        embeddings = self.encoder_image(x_batch)  # Image embeddings, or full joint representation can be used
        return embeddings
