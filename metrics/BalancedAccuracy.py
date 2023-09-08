import pandas as pd

from metrics.Metric import Metric


class BalancedAccuracy(Metric):

    def __init__(self):
        name = "BalancedACC"
        super().__init__(name)
        self.res = []

    def update(self, y_true, y_pred, s):
        res = self.calculate(y_true, y_pred, s)
        self.res.append(res)

        return res

    def calculate(self, y_true, y_pred, s):
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


def divide_one(a, b):
    if b == 0:
        return 0
    return a / b
