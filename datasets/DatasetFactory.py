from datasets.fairMNIST.FairMNIST import FairMNIST


def get_datasets():
    return [FairMNIST()]


def get_dataset_by_name(name):
    for dataset in get_datasets():
        if name == dataset.name:
            return dataset

    raise Exception("No Dataset with the name ", name)

