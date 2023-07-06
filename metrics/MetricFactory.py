from metrics.Accuracy import Accuracy
from metrics.BalancedAccuracy import BalancedAccuracy


def get_metrics(is_image):
    if not is_image:
        return [Accuracy(), BalancedAccuracy()]
    else:
        return [Accuracy(), BalancedAccuracy()]
