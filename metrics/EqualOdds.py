import pandas as pd

from metrics.Metric import Metric


class EqualOdds(Metric):

    def __init__(self):
        name = "EQO"
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
        true_positive_priv = len(
            df[(df["y_pred"] == 1) & (df["y"] == 1) & (df["s"] == 1)]
        )
        total_priv = len(df[(df["y"] == 1) & (df["s"] == 1)])
        true_positive_unpriv = len(
            df[(df["y_pred"] == 1) & (df["y"] == 1) & (df["s"] == 0)]
        )
        total_unpriv = len(df[(df["y"] == 1) & (df["s"] == 0)])
        res = divide(divide(true_positive_unpriv, total_unpriv), divide(true_positive_priv, total_priv))
        if res > 1:
            res = 1 / res

        false_positive_priv = len(
            df[(df["y_pred"] == 1) & (df["y"] == 0) & (df["s"] == 1)]
        )
        total_priv_2 = len(df[(df["y"] == 0) & (df["s"] == 1)])
        false_positive_unpriv = len(
            df[(df["y_pred"] == 1) & (df["y"] == 0) & (df["s"] == 0)]
        )
        total_unpriv_2 = len(df[(df["y"] == 0) & (df["s"] == 0)])
        res2 = divide(divide(false_positive_unpriv, total_unpriv_2), divide(false_positive_priv, total_priv_2))
        if res2 > 1:
            res2 = 1 / res2

        final_res = (res + res2) / 2

        return final_res


def divide(a, b):
    if b == 0:
        return 0
    return a / b


def divide_one(a, b):
    if b == 0:
        return 0
    return a / b
