import logging
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, TimeDistributed, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model, Sequential


class AssignmentModel:
    def __init__(self, seed):
        initializer = tf.keras.initializers.RandomNormal(seed=seed)
        self.n_rounds = 1
        self.batch_size = 1  # TODO
        self.n_epochs = 10
        self.initializer = initializer
        self.num_pairs = 100
        self.image_shape = (28, 28, 1)  # Image input shape (28x28 grayscale)
        self.label_shape = (10,)  # One-hot encoded label of size 10
        self.num_classes = 2  # Number of classes
        self._initialize_model()

    def _initialize_model(self):
        # Input layers
        input_images = Input(shape=(self.num_pairs, *self.image_shape), name="input_images")
        input_labels = Input(shape=(self.num_pairs, *self.label_shape), name="input_labels")

        # Shared CNN for image feature extraction
        image_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu')
        ], name="image_feature_extractor")

        # Process each image through the CNN
        images_reshaped = TimeDistributed(image_model)(input_images)

        # Process labels through a simple dense layer
        label_model = Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu')
        ], name="label_feature_extractor")

        # Process each label
        labels_reshaped = TimeDistributed(label_model)(input_labels)

        # Concatenate image and label features for each pair
        combined_features = Concatenate()([images_reshaped, labels_reshaped])

        # Transformer-based encoding for the pairs
        attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(combined_features, combined_features)
        encoded_features = LayerNormalization()(attention_output + combined_features)
        encoded_features = Dense(256, activation='relu')(encoded_features)

        # Aggregate the encoded features (e.g., through a fully connected network)
        final_representation = Flatten()(encoded_features)

        # Final classification layer
        output = Dense(self.num_classes, activation='softmax', name="model_selector_output")(final_representation)

        # Build and compile the model
        self.model = Model(inputs=[input_images, input_labels], outputs=output, name="meta_learner")

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def learn(self, x_image_train, x_label_train, y_train):
        tf.keras.utils.disable_interactive_logging()
        logging.info(f"x_image_train: {len(x_image_train)}, x_label_train: {len(x_label_train)}, y_train: {len(y_train)}")
        logging.info(x_image_train[0][0].shape)  # this is one image
        logging.info(x_label_train[0][0].shape)  # this is one-hot encoding
        logging.info(y_train[0])  # this is one-hot encoding
        logging.info(f"x_image_train shape: {x_image_train.shape}")
        logging.info(f"x_label_train shape: {x_label_train.shape}")
        logging.info(f"y_train shape: {y_train.shape}")
        self.model.fit(
            [x_image_train, x_label_train], y_train, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0
        )

    def predict(self, x_image, x_label):
        logging.info(f"x_image shape: {x_image.shape}")
        logging.info(f"x_label shape: {x_label.shape}")
        return self.model.predict([x_image, x_label])

    def add_class(self):
        self.num_classes += 1
        new_output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        self.model.pop()  # Remove the last layer (output layer)
        self.model.add(new_output_layer)  # Add the new output layer

        old_weights = self.model.get_weights()
        new_weights = self.model.get_weights()
        for i in range(len(new_weights) - 2):  # Exclude the output layer weights and biases
            new_weights[i] = old_weights[i]
        new_weights[-2][:, :-1] = old_weights[-2][:, :]  # Copy weights for the existing classes
        # Randomly initialize the new weights for the new class (last column in the weight matrix)
        new_weights[-2][:, -1] = tf.random.normal(shape=(old_weights[-2].shape[0],))
        new_weights[-1][:-1] = old_weights[-1]  # Preserve old biases
        new_weights[-1][-1] = 0  # Set the new bias to zero for the new class
        self.model.set_weights(new_weights)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
