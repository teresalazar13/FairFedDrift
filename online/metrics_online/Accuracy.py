from metrics_online.Metric import Metric


class Accuracy(Metric):

    def __init__(self):
        name = "ACC"
        super().__init__(name)
        self.correct = []
        self.res = []
        self.window = 1000

    def update(self, y_true, y_pred, _):
        if y_true == y_pred:
            self.correct.append(1)
        else:
            self.correct.append(0)
        if len(self.correct) <= self.window:
            res = sum(self.correct) / len(self.correct)
        else:
            res = sum(self.correct[len(self.correct)-self.window:]) / self.window
        self.res.append(res)

        return res
