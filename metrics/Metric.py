from abc import abstractmethod


class Metric:
    def __init__(self, name):
        self.name = name
        self.res = []

    def update(self, y_true, y_pred, s, metrics_evaluation):
        res = self.calculate(y_true, y_pred, s, metrics_evaluation)
        self.res.append(res)

        return res

    @abstractmethod
    def calculate(self, y_true, y_pred, s, _):
        raise NotImplementedError("Must implement calculate")
