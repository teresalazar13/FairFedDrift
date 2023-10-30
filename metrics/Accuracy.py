import pandas as pd

from metrics.Metric import Metric


class Accuracy(Metric):

    def __init__(self):
        name = "ACC"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred

        return len(df[df["y"] == df["y_pred"]]) / len(df)
