from metrics.Accuracy import Accuracy
from metrics.BalancedAccuracy import BalancedAccuracy
from metrics.EqualOdds import EqualOdds
from metrics.EqualOpportunity import EqualOpportunity
from metrics.Loss import Loss
from metrics.StatisticalParity import StatisticalParity


def get_metrics(is_binary_target):
    if is_binary_target:
        return [Accuracy(), Loss(), BalancedAccuracy(), StatisticalParity(), EqualOpportunity(), EqualOdds()]
    else:
        return [Accuracy(), Loss(), BalancedAccuracy()]


def get_all_metrics():
    return [Accuracy(), Loss(), BalancedAccuracy(), StatisticalParity(), EqualOpportunity(), EqualOdds()]


def get_metrics_by_names(names):
    metrics = []
    for metric in get_all_metrics():
        for name in names:
            if name == metric.name:
                metrics.append(metric)

    if len(metrics) == 0:
        raise Exception("No Metrics with the names ", names)

    return metrics
