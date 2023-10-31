import pandas as pd

from metrics.Metric import Metric


class EqualOpportunity(Metric):

    def __init__(self):
        name = "EO"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __, s):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred
        df["s"] = s
        true_positive_priv = len(
            df[(df["y_pred"] == 1) & (df["y"] == 1) & (df["s"] == 1)]
        )
        total_priv = len(
            df[(df["y"] == 1) & (df["s"] == 1)]
        )
        true_positive_unpriv = len(
            df[(df["y_pred"] == 1) & (df["y"] == 1) & (df["s"] == 0)]
        )
        total_unpriv = len(
            df[(df["y"] == 1) & (df["s"] == 0)]
        )

        res = divide(divide(true_positive_unpriv, total_unpriv), divide(true_positive_priv, total_priv))

        if res > 1:
            res = 1 / res

        return res


def divide(a, b):
    if b == 0:
        return 0
    return a / b


def divide_one(a, b):
    if b == 0:
        return 0
    return a / b
