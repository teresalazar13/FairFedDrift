from federated.nn_pt import NNPT
from federated.nn_tf_2 import NNTF_2
from federated.nn_tf import NNTF


def NN_model(dataset, seed):
    if dataset.is_pt:
        return NNPT(dataset, seed)
        #return NNTF_2(dataset, seed)
    else:
        return NNTF(dataset, seed)
