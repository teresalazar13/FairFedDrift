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

        res_list = []
        y_to_go = set(y_true)
        if len(set(y_true)) == 2:
            y_to_go = [1]
        for y in y_to_go:
            true_positive = len(df[(df["y_true"] == y) & (df["y_pred"] == y)])
            false_positive = len(df[(df["y_true"] != y) & (df["y_pred"] == y)])
            false_negative = len(df[(df["y_true"] == y) & (df["y_pred"] != y)])

            if true_positive == 0:
                res_list.append(0)
            else:
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)
                f1_score = 2 * (precision * recall) / (precision + recall)
                res_list.append(f1_score)

        return sum(res_list) / len(res_list)
