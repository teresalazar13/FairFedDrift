from datasets.tabular.adult.Adult import Adult
from datasets.tabular.dutch.Dutch import Dutch
from datasets.image.fairMNIST.FairMNIST import FairMNIST
from datasets.image.fairFashionMNIST.FairFashionMNIST import FairFashionMNIST
from datasets.synthetic.Synthetic import Synthetic


def get_datasets():
    return [FairMNIST(), FairFashionMNIST(), Synthetic(), Dutch(), Adult()]


def get_dataset_by_name(name):
    for dataset in get_datasets():
        if name == dataset.name:
            return dataset

    raise Exception("No Dataset with the name ", name)
