from metrics_online.Metric import Metric


class EqualizedOdds(Metric):

    def __init__(self):
        name = "EQO"
        super().__init__(name)

        self.positive_privileged_true = 0
        self.positive_unprivileged_true = 0
        self.positive_privileged = 0
        self.positive_unprivileged = 0

        self.negative_privileged_true = 0
        self.negative_unprivileged_true = 0
        self.negative_privileged = 0
        self.negative_unprivileged = 0

        self.res = []

    def update(self, y_true, y_pred, s):
        if y_true == 1:
            if s == 1:
                self.positive_privileged += 1
                if y_pred == 1:
                    self.positive_privileged_true += 1
            else:
                self.positive_unprivileged += 1
                if y_pred == 1:
                    self.positive_unprivileged_true += 1
        else:
            if s == 1:
                self.negative_privileged += 1
                if y_pred == 0:
                    self.negative_privileged_true += 1
            else:
                self.negative_unprivileged += 1
                if y_pred == 0:
                    self.negative_unprivileged_true += 1

        if self.positive_unprivileged != 0 and self.positive_privileged \
                and self.positive_privileged_true / self.positive_privileged != 0:
            eo = (self.positive_unprivileged_true / self.positive_unprivileged) \
                 / (self.positive_privileged_true / self.positive_privileged)
            if eo > 1:
                eo = 1 / eo
        else:
            eo = 0

        if self.negative_unprivileged != 0 and self.negative_privileged \
                and self.negative_privileged_true / self.negative_privileged != 0:
            fpr = (self.negative_unprivileged_true / self.negative_unprivileged) \
                 / (self.negative_privileged_true / self.negative_privileged)
            if fpr > 1:
                fpr = 1 / fpr
        else:
            fpr = 0

        eqo = (fpr + eo) / 2

        self.res.append(eqo)
        return eqo
