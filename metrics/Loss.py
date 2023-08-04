import tensorflow as tf
from metrics.Metric import Metric


class Loss(Metric):

    def __init__(self):
        name = "Loss"
        super().__init__(name)
        self.res = []

    def update(self, y_true_original, y_pred_original, _, __, ___):
        res = self.calculate(y_true_original, y_pred_original, None, None, None)
        self.res.append(res)

        return res

    def calculate(self, y_true_original, y_pred_original, _, __, ___):
        cce = tf.keras.losses.CategoricalCrossentropy()

        return cce(y_true_original, y_pred_original).numpy()
