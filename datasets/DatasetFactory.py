from datasets.tabular.Adult_GDrift.Adult_GDrift import Adult_GDrift
from datasets.image.MNIST_GDrift.MNIST_GDrift import MNIST_GDrift
from datasets.image.FashionMNIST_GDrift.FashionMNIST_GDrift import FashionMNIST_GDrift


def get_datasets():
    return [MNIST_GDrift(), FashionMNIST_GDrift(), Adult_GDrift()]


def get_dataset_by_name(name):
    for dataset in get_datasets():
        if name == dataset.name:
            return dataset

    raise Exception("No Dataset with the name ", name)
