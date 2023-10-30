from metrics.Metric import Metric


class Loss(Metric):

    def __init__(self):
        name = "Loss"
        super().__init__(name)

    def calculate(self, _, __, ___, metrics_evaluation):  # contains loss according to model.py
        return metrics_evaluation
