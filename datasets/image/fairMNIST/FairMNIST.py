from keras.datasets import mnist
from datasets.image.ImageDataset import ImageDataset


class FairMNIST(ImageDataset):

    def __init__(self):
        name = "fairMNIST"
        (train_X_priv, train_y_priv), (test_X_priv, test_y_priv) = mnist.load_data()
        super().__init__(name, train_X_priv, train_y_priv, test_X_priv, test_y_priv)
