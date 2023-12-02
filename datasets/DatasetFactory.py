from datasets.image.CIFAR_GDrift.CIFAR_GDrift import CIFAR_GDrift
from datasets.image.CelebA_GDrift.CelebA_GDrift import CelebA_GDrift
from datasets.tabular.Adult_GDrift.Adult_GDrift import Adult_GDrift
from datasets.tabular.dutch.Dutch import Dutch
from datasets.image.MNIST_GDrift.MNIST_GDrift import MNIST_GDrift
from datasets.image.FashionMNIST_GDrift.FashionMNIST_GDrift import FashionMNIST_GDrift
from datasets.synthetic.Synthetic import Synthetic


def get_datasets():
    return [MNIST_GDrift(), FashionMNIST_GDrift(), CelebA_GDrift(), CIFAR_GDrift(), Synthetic(), Adult_GDrift(), Dutch()]


def get_dataset_by_name(name):
    for dataset in get_datasets():
        if name == dataset.name:
            return dataset

    raise Exception("No Dataset with the name ", name)
