from federated.nn_pt import NNPT
from federated.nn_tf import NNTF


def NN_model(dataset, seed):
    if dataset.is_pt:
        return NNPT(dataset, seed)
    else:
        return NNTF(dataset, seed)
