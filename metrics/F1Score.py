import pandas as pd

from metrics.Metric import Metric


class F1Score(Metric):

    def __init__(self):
        name = "F1Score"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __, ___):
        df = pd.DataFrame()
        df["y_true"] = y_true
        df["y_pred"] = y_pred

        true_positive = len(df[(df["y_true"] == 1) & (df["y_pred"] == 1)])
        false_positive = len(df[(df["y_true"] == 0) & (df["y_pred"] == 1)])
        false_negative = len(df[(df["y_true"] == 1) & (df["y_pred"] == 0)])

        if true_positive == 0:
            return 0.0

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score
