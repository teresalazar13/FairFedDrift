import pandas as pd

from metrics.Metric import Metric


class BalancedAccuracy(Metric):

    def __init__(self):
        name = "BalancedACC"
        super().__init__(name)
        self.res = []

    def update(self, _, __, y_true, y_pred, s):
        res = self.calculate(None, None, y_true, y_pred, s)
        self.res.append(res)

        return res

    def calculate(self, _, __, y_true, y_pred, s):
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
        acc_priv = correct_priv / total_priv
        acc_unpriv = correct_unpriv / total_unpriv
        if acc_priv != 0:
            res = acc_unpriv / acc_priv
        else:
            res = 0
        if res > 1:
            res = 1 / res

        return res
