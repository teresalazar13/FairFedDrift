from metrics.Accuracy import Accuracy
from metrics.BalancedAccuracy import BalancedAccuracy


def get_metrics(is_image):
    if not is_image:
        return [Accuracy(), BalancedAccuracy()]
    else:
        return [Accuracy(), BalancedAccuracy()]


def get_all_metrics():
    return [Accuracy(), BalancedAccuracy()]


def get_metrics_by_names(names):
    metrics = []
    for metric in get_all_metrics():
        for name in names:
            if name == metric.name:
                metrics.append(metric)

    if len(metrics) == 0:
        raise Exception("No Metrics with the names ", names)

    return metrics
