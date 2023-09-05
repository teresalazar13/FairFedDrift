import pandas as pd

from metrics.Metric import Metric


class StatisticalParity(Metric):

    def __init__(self):
        name = "SP"
        super().__init__(name)
        self.res = []

    def update(self, _, __, y_true, y_pred, s):
        res = self.calculate(None, None, y_true, y_pred, s)
        self.res.append(res)

        return res

    def calculate(self, _, __, y_true, y_pred, s):
        df = pd.DataFrame()
        df["y_pred"] = y_pred
        df["s"] = s
        positive_priv = len(
            df[(df["y_pred"] == 1) & (df["s"] == 1)]
        )
        total_priv = len(df[df["s"] == 1])
        positive_unpriv = len(
            df[(df["y_pred"] == 1) & (df["s"] == 0)]
        )
        total_unpriv = len(df[df["s"] == 0])

        res = divide(divide(positive_unpriv, total_unpriv), divide(positive_priv, total_priv))

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
