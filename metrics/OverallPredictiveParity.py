import pandas as pd
import logging

from metrics.Metric import Metric


class OverallPredictiveParity(Metric):

    def __init__(self):
        name = "OPP"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __, s):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred
        df["s"] = s

        res_list = []

        for y in set(y_true):
            correct_priv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 1) & (df["y"] == y)]
            )
            total_priv = len(
                df[(df["s"] == 1) & (df["y_pred"] == y)]
            )
            correct_unpriv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 0) & (df["y"] == y)]
            )
            total_unpriv = len(
                df[(df["s"] == 0) & (df["y_pred"] == y)]
            )

            acc_priv = divide(correct_priv, total_priv)
            acc_unpriv = divide(correct_unpriv, total_unpriv)
            res = divide(acc_unpriv, acc_priv)

            if res > 1:
                res = 1 / res

            if total_unpriv != 0 and total_priv != 0:
                res_list.append(res)

            logging.info("{} {}".format(y, res))

        return sum(res_list) / len(res_list)


def divide(a, b):
    if b == 0:
        return 0
    return a / b
