import numpy as np
from keras.datasets import mnist
from datasets.image.ImageDataset import ImageDataset


class MNIST_GDrift(ImageDataset):

    def __init__(self):
        name = "MNIST-GDrift"
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        input_shape = (28, 28, 1)
        is_large = False
        is_binary_target = False
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        super().__init__(name, input_shape, is_large, is_binary_target, X, y)
