from metrics_online.Metric import Metric


class StatisticalParity(Metric):

    def __init__(self):
        name = "SP"
        super().__init__(name)

        self.positive_privileged = 0
        self.positive_unprivileged = 0
        self.privileged = 0
        self.unprivileged = 0

        self.res = []

    def update(self, _, y_pred, s):
        if s == 1:
            self.privileged += 1
            if y_pred == 1:
                self.positive_privileged += 1
        else:
            self.unprivileged += 1
            if y_pred == 1:
                self.positive_unprivileged += 1

        if self.unprivileged != 0 and self.privileged != 0 and (self.positive_privileged / self.privileged) != 0:
            sp = (self.positive_unprivileged / self.unprivileged) / (self.positive_privileged / self.privileged)
            if sp > 1:
                sp = 1 / sp
        else:
            sp = 0

        self.res.append(sp)

        return sp
