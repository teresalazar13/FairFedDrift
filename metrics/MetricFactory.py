from metrics.Accuracy import Accuracy
from metrics.AccuracyEquality import AccuracyEquality
from metrics.Loss import Loss
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.MinimumEqualityOpportunity import MinimumEqualityOpportunity
from metrics.MinimumPredictiveParity import MinimumPredictiveParity
from metrics.OverallEqualityOpportunity import OverallEqualityOpportunity
from metrics.OverallPredictiveParity import OverallPredictiveParity


def get_metrics(_):
    return [
        Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(),
        AccuracyEquality(), OverallEqualityOpportunity(), OverallPredictiveParity(), MinimumEqualityOpportunity(),
        MinimumPredictiveParity(),
    ]


def get_metric_by_names(name):
    for metric in get_metrics(None):
        if name == metric.name:
            return metric

    raise Exception("No Metric with the name ", name)
