import numpy as np
from keras.datasets import cifar100
from datasets.image.ImageDataset import ImageDataset


class CIFAR100_GDrift(ImageDataset):

    def __init__(self):
        name = "CIFAR100-GDrift"
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()
        input_shape = (3, 224, 224)  # PyTorch uses (C, H, W) format
        is_pt = True
        n_classes = 100
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        super().__init__(name, input_shape, is_pt, n_classes, X, y)
