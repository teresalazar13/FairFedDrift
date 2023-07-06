from metrics_online.Accuracy import Accuracy
from metrics_online.BalancedAccuracy import BalancedAccuracy
from metrics_online.EqualityOpportunity import EqualityOpportunity
from metrics_online.EqualizedOdds import EqualizedOdds
from metrics_online.F1Score import F1Score
from metrics_online.StatisticalParity import StatisticalParity


def get_metrics(is_image):
    if not is_image:
        return [Accuracy(), F1Score(), StatisticalParity(), EqualityOpportunity(), EqualizedOdds(), BalancedAccuracy()]
    else:
        return [Accuracy(), BalancedAccuracy()]
