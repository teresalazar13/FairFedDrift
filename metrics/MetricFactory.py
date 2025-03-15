from metrics.Accuracy import Accuracy
from metrics.AccuracyEquality import AccuracyEquality
from metrics.Loss import Loss
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.OverallEqualityOpportunity import OverallEqualityOpportunity
from metrics.OverallPredictiveParity import OverallPredictiveParity
from metrics.GroupTPR import GroupTPR
from metrics.GroupPPV import GroupPPV


def get_metrics(dataset_is_pt):
    if dataset_is_pt:
        return [
            Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(), AccuracyEquality(),
            GroupTPR(), GroupPPV()
        ]
    else:
        return [
            Accuracy(), Loss(), LossPrivileged(), LossUnprivileged(), AccuracyEquality(),
            OverallEqualityOpportunity(), OverallPredictiveParity()
        ]
