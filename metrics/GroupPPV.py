import pandas as pd
import logging

from metrics.Metric import Metric


class GroupPPV(Metric):

    def __init__(self):
        name = "OPP"
        super().__init__(name)

    def calculate(self, y_true, y_pred, _, __, s):
        df = pd.DataFrame()
        df["y"] = y_true
        df["y_pred"] = y_pred
        df["s"] = s
        ppvs_priv = []
        ppvs_unpriv = []

        for y in set(y_true):
            correct_priv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 1) & (df["y"] == y)]
            )
            total_priv = len(
                df[(df["s"] == 1) & (df["y_pred"] == y)]
            )
            t_priv = {
                df[(df["s"] == 1) & (df["y"] == y)]
            }
            correct_unpriv = len(
                df[(df["y"] == df["y_pred"]) & (df["s"] == 0) & (df["y"] == y)]
            )
            total_unpriv = len(
                df[(df["s"] == 0) & (df["y_pred"] == y)]
            )
            t_unpriv = {
                df[(df["s"] == 0) & (df["y"] == y)]
            }

            if t_priv != 0:
                ppv_priv = divide(correct_priv, total_priv)
                logging.info("S=1 {} - {}".format(y, ppv_priv))
                ppvs_priv.append(ppv_priv)
            if t_unpriv != 0:
                ppv_unpriv = divide(correct_unpriv, total_unpriv)
                logging.info("S=0 {} - {}".format(y, ppv_unpriv))
                ppvs_unpriv.append(ppv_unpriv)

        avg_ppvs_priv = sum(ppvs_priv) / len(ppvs_priv)
        avg_ppvs_unpriv = sum(ppvs_unpriv) / len(ppvs_unpriv)
        res = divide(avg_ppvs_priv, avg_ppvs_unpriv)
        if res > 1:
            res = 1 / res

        return res


def divide(a, b):
    if b == 0:
        return 0
    return a / b
