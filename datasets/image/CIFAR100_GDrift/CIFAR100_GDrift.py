import numpy as np
import tensorflow as tf
from keras.datasets import cifar100
from datasets.image.ImageDataset import ImageDataset


class CIFAR100_GDrift(ImageDataset):

    def __init__(self):
        name = "CIFAR100-GDrift"
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()
        # Resize images to match ResNet18's 224x224 input size
        train_X = tf.image.resize(train_X, (224, 224)) / 255.0
        test_X = tf.image.resize(test_X, (224, 224)) / 255.0
        input_shape = (224, 224, 3)
        is_large = True
        is_binary_target = False
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        super().__init__(name, input_shape, is_large, is_binary_target, X, y)
