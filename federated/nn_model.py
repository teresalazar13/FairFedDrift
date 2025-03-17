from federated.nn_pt import NNPT
from federated.nn_pt_2 import NNPT2
from federated.nn_pt_3 import NNPT3
from federated.nn_tf_2 import NNTF_2
from federated.nn_tf import NNTF


def NN_model(dataset, seed):
    if dataset.is_pt:
        return NNPT3(dataset, seed)
        #return NNPT2(dataset, seed)
        #return NNPT(dataset, seed)
        #return NNTF_2(dataset, seed)
    else:
        return NNTF(dataset, seed)
