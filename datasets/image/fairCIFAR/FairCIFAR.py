from keras.datasets import cifar10
from datasets.image.ImageDataset import ImageDataset


class FairCIFAR(ImageDataset):

    def __init__(self):
        name = "fairCIFAR"
        (train_X_priv, train_y_priv), (test_X_priv, test_y_priv) = cifar10.load_data()
        input_shape = (32, 32, 3)

        super().__init__(name, input_shape, train_X_priv, train_y_priv, test_X_priv, test_y_priv)
