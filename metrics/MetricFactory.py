from metrics.Accuracy import Accuracy
from metrics.BalancedAccuracy import BalancedAccuracy
from metrics.EqualOdds import EqualOdds
from metrics.EqualOpportunity import EqualOpportunity
from metrics.Loss import Loss
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.StatisticalParity import StatisticalParity


def get_metrics(is_binary_target):
    if is_binary_target:
        #return [Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(), BalancedAccuracy(), StatisticalParity(), EqualOpportunity(), EqualOdds()]
        return [Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(), BalancedAccuracy()]
    else:
        return [Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(), BalancedAccuracy()]


def get_all_metrics():
    return [Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(), BalancedAccuracy(), StatisticalParity(), EqualOpportunity(), EqualOdds()]


def get_metric_by_names(name):
    for metric in get_all_metrics():
        if name == metric.name:
            return metric

    raise Exception("No Metric with the name ", name)
