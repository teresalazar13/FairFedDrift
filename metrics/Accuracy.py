import pandas as pd

from metrics.Metric import Metric


class Accuracy(Metric):

    def __init__(self):
        name = "ACC"
        super().__init__(name)
        self.res = []

    def update(self, _, __, y_true, y_pred, ___):
        res = self.calculate(None, None, y_true, y_pred, None)
        self.res.append(res)

        return res

    def calculate(self, _, __, y_true, y_pred, ___):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred

        return len(df[df["y"] == df["y_pred"]]) / len(df)
