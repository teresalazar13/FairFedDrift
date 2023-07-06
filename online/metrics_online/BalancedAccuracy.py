from metrics_online.Metric import Metric


class BalancedAccuracy(Metric):

    def __init__(self):
        name = "BalancedACC"
        super().__init__(name)
        self.correct_priv = []
        self.correct_unpriv = []
        self.res = []
        self.window = 1000

    def update(self, y_true, y_pred, s):
        if s == 1:
            if y_true == y_pred:
                self.correct_priv.append(1)
            else:
                self.correct_priv.append(0)
        else:
            if y_true == y_pred:
                self.correct_unpriv.append(1)
            else:
                self.correct_unpriv.append(0)

        if len(self.correct_priv) != 0 and len(self.correct_unpriv) != 0:
            if len(self.correct_priv) < self.window:
                acc_priv = sum(self.correct_priv) / len(self.correct_priv)
                acc_unpriv = sum(self.correct_unpriv) / len(self.correct_unpriv)
            else:
                acc_priv = sum(self.correct_priv[len(self.correct_priv)-self.window:]) / self.window
                acc_unpriv = sum(self.correct_unpriv[len(self.correct_unpriv)-self.window:]) / self.window

            if acc_priv == 0:
                res = 0
            else:
                res = acc_unpriv / acc_priv
                if res > 1:
                    res = 1 / res

        else:
            res = 0

        self.res.append(res)

        return res
