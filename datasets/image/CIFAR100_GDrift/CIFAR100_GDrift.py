import numpy as np
import tensorflow as tf
from keras.datasets import cifar100
from tensorflow.data import AUTOTUNE
from datasets.image.ImageDataset import ImageDataset


class CIFAR100_GDrift(ImageDataset):
    def __init__(self):
        name = "CIFAR100-GDrift"
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()
        train_X, test_X = train_X / 255.0, test_X / 255.0
        input_shape = (224, 224, 3)
        is_large = True
        is_binary_target = False

        # Create tf.data.Dataset pipelines to handle resizing efficiently
        self.train_dataset = (tf.data.Dataset.from_tensor_slices((train_X, train_y))
                              .map(self.preprocess_image, num_parallel_calls=AUTOTUNE)
                              .batch(32)
                              .prefetch(AUTOTUNE))

        self.test_dataset = (tf.data.Dataset.from_tensor_slices((test_X, test_y))
                             .map(self.preprocess_image, num_parallel_calls=AUTOTUNE)
                             .batch(32)
                             .prefetch(AUTOTUNE))

        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)

        super().__init__(name, input_shape, is_large, is_binary_target, X, y)

    @staticmethod
    def preprocess_image(image, label):
        """Resize images to 224x224 on the fly."""
        image = tf.image.resize(image, (224, 224))
        return image, label
