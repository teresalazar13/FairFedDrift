from keras.datasets import fashion_mnist
from datasets.image.ImageDataset import ImageDataset


class FairFashionMNIST(ImageDataset):

    def __init__(self):
        name = "fairFashionMNIST"
        (train_X_priv, train_y_priv), (test_X_priv, test_y_priv) = fashion_mnist.load_data()
        input_shape = (28, 28, 1)

        super().__init__(name, input_shape, train_X_priv, train_y_priv, test_X_priv, test_y_priv)
