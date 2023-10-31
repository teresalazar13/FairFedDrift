from abc import abstractmethod


class Metric:
    def __init__(self, name):
        self.name = name
        self.res = []

    def update(self, y_true, y_pred, y_true_raw, y_pred_raw, s):
        res = self.calculate(y_true, y_pred, y_true_raw, y_pred_raw, s)
        self.res.append(res)

        return res

    @abstractmethod
    def calculate(self, y_true, y_pred, y_true_raw, y_pred_raw, s):
        raise NotImplementedError("Must implement calculate")
