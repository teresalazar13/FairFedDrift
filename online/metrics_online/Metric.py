from abc import abstractmethod


class Metric:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def update(self, y_true, y_pred, s):
        raise NotImplementedError("Must implement update")
