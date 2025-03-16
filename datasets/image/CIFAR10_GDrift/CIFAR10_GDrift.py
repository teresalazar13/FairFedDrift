import numpy as np
from keras.datasets import cifar10
from datasets.image.ImageDataset import ImageDataset


class CIFAR10_GDrift(ImageDataset):

    def __init__(self):
        name = "CIFAR10-GDrift"
        (train_X, train_y), (test_X, test_y) = cifar10.load_data()
        input_shape = (3, 224, 224)
        is_pt = True
        n_classes = 10
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        super().__init__(name, input_shape, is_pt, n_classes, X, y)
