from datasets.adult.Adult import Adult
from datasets.dutch.Dutch import Dutch
from datasets.fairMNIST.FairMNIST import FairMNIST
from datasets.synthetic.Synthetic import Synthetic


def get_datasets():
    return [FairMNIST(), Synthetic(), Dutch(), Adult()]


def get_dataset_by_name(name):
    for dataset in get_datasets():
        if name == dataset.name:
            return dataset

    raise Exception("No Dataset with the name ", name)
