import pandas as pd

from metrics.Metric import Metric


class Loss(Metric):

    def __init__(self):
        name = "Loss"
        super().__init__(name)
        self.res = []

    def update(self, y_true, y_pred, _):
        res = self.calculate(y_true, y_pred, _)
        self.res.append(res)

        return res

    def calculate(self, y_true, y_pred, _):
        print(y_true)
        print(y_pred)
        exit()
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred

        return len(df[df["y"] == df["y_pred"]]) / len(df)
