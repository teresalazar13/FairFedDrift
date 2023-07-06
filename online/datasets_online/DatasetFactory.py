from datasets_online.adult.Adult import Adult
from datasets_online.compas.Compas import Compas
from datasets_online.fairMNIST.FairMNIST import FairMNIST
from datasets_online.synthetic.SyntheticDataset import SyntheticDataset


def get_datasets():
    return [SyntheticDataset(), FairMNIST(), Adult(), Compas()]

def get_dataset_by_name(name):
    for dataset in get_datasets():
        if name == dataset.name:
            return dataset

    raise Exception("No Dataset with the name ", name)

