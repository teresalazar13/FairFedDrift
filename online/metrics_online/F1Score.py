from metrics_online.Metric import Metric


class F1Score(Metric):

    def __init__(self):
        name = "F1Score"
        super().__init__(name)
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.res = []

    def update(self, y_true, y_pred, _):
        if y_true == 1 and y_pred == 1:
            self.TP += 1
        elif y_true == 0 and y_pred == 1:
            self.FP += 1
        elif y_true == 1 and y_pred == 0:
            self.FN += 1

        if 2 * self.TP + self.FP + self.FN != 0:
            value = (2 * self.TP) / (2 * self.TP + self.FP + self.FN)
        else:
            value = 0
        self.res.append(value)

        return value
