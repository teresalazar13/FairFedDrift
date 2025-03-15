import pandas as pd
import logging

from metrics.Metric import Metric


class OverallEqualityOpportunity(Metric):

    def __init__(self):
        name = "OEQ"
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
                df[(df["s"] == 1) & (df["y"] == y)]
            )
            correct_unpriv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 0) & (df["y"] == y)]
            )
            total_unpriv = len(
                df[(df["s"] == 0) & (df["y"] == y)]
            )

            acc_priv = divide(correct_priv, total_priv)
            acc_unpriv = divide(correct_unpriv, total_unpriv)
            res = divide(acc_unpriv, acc_priv)

            if res > 1:
                res = 1 / res

            if total_unpriv != 0 and total_priv != 0:
                res_list.append(res)
                logging.info("{} - {}".format(y, res))

        if res_list != []:
            return sum(res_list) / len(res_list)

        tprs_priv = []
        tprs_unpriv = []
        for y in set(y_true):
            correct_priv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 1) & (df["y"] == y)]
            )
            total_priv = len(
                df[(df["s"] == 1) & (df["y"] == y)]
            )
            correct_unpriv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 0) & (df["y"] == y)]
            )
            total_unpriv = len(
                df[(df["s"] == 0) & (df["y"] == y)]
            )

            if total_priv != 0:
                tpr_priv = divide(correct_priv, total_priv)
                logging.info("S=1 {} - {}".format(y, tpr_priv))
                tprs_priv.append(tpr_priv)
            if total_unpriv != 0:
                tpr_unpriv = divide(correct_unpriv, total_unpriv)
                logging.info("S=0 {} - {}".format(y, tprs_unpriv))
                tprs_unpriv.append(tpr_unpriv)

        avg_tprs_priv = sum(tprs_priv) / len(tprs_priv)
        avg_tprs_unpriv = sum(tprs_unpriv) / len(tprs_unpriv)
        res = divide(avg_tprs_priv, avg_tprs_unpriv)
        if res > 1:
            res = 1 / res

        return res


def divide(a, b):
    if b == 0:
        return 0
    return a / b
