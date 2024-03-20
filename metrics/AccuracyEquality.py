import pandas as pd

from metrics.Metric import Metric


class AccuracyEquality(Metric):

    def __init__(self):
        name = "AEQ"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __, s):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred
        df["s"] = s
        correct_priv = len(
            df[(df["y"] == df["y_pred"]) & (df["s"] == 1)]
        )
        total_priv = len(df[df["s"] == 1])
        correct_unpriv = len(
            df[(df["y"] == df["y_pred"]) & (df["s"] == 0)]
        )
        total_unpriv = len(df[df["s"] == 0])

        acc_priv = divide(correct_priv, total_priv)
        acc_unpriv = divide(correct_unpriv, total_unpriv)
        res = divide(acc_unpriv, acc_priv)

        if res > 1:
            res = 1 / res

        return res


def divide(a, b):
    if b == 0:
        return 0
    return a / b
