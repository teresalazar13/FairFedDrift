import pandas as pd
from metrics.Metric import Metric


class F1ScoreEquality(Metric):

    def __init__(self):
        name = "F1 Score Equality"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __, s):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred
        df["s"] = s

        res_list = []
        y_to_go = set(y_true)
        if len(set(y_true)) == 2:
            y_to_go = [1]
        for y in y_to_go:
            true_positive_priv = len(df[(df["y"] == y) & (df["y_pred"] == y) & (df["s"] == 1)])
            false_positive_priv = len(df[(df["y"] != y) & (df["y_pred"] == y) & (df["s"] == 1)])
            false_negative_priv = len(df[(df["y"] == y) & (df["y_pred"] != y) & (df["s"] == 1)])

            precision_priv = divide(true_positive_priv, true_positive_priv + false_positive_priv)
            recall_priv = divide(true_positive_priv, true_positive_priv + false_negative_priv)

            f1_priv = calculate_f1(precision_priv, recall_priv)

            true_positive_unpriv = len(df[(df["y"] == y) & (df["y_pred"] == y) & (df["s"] == 0)])
            false_positive_unpriv = len(df[(df["y"] != y) & (df["y_pred"] == y) & (df["s"] == 0)])
            false_negative_unpriv = len(df[(df["y"] == y) & (df["y_pred"] != y) & (df["s"] == 0)])

            precision_unpriv = divide(true_positive_unpriv, true_positive_unpriv + false_positive_unpriv)
            recall_unpriv = divide(true_positive_unpriv, true_positive_unpriv + false_negative_unpriv)

            f1_unpriv = calculate_f1(precision_unpriv, recall_unpriv)

            print(f1_unpriv, f1_priv)
            res = divide(f1_unpriv, f1_priv)

            if res > 1:
                res = 1 / res

            res_list.append(res)

        return sum(res_list) / len(res_list)


def divide(a, b):
    if b == 0:
        return 0
    return a / b


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
