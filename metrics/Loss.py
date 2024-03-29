from metrics.Metric import Metric

import tensorflow as tf


class Loss(Metric):

    def __init__(self):
        name = "Loss"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):
        if y_pred_raw.shape[1] == 1:
            # Binary classification problem (y_pred_raw has shape (batch_size, 1))
            y_pred_raw_reshaped = tf.reshape(y_pred_raw, [len(y_pred_raw)])
            #logging.info(tf.shape(y_true_raw))
            #logging.info(tf.shape(y_pred_raw_reshaped))
            loss = tf.keras.losses.binary_crossentropy(y_true_raw, y_pred_raw_reshaped)
        else:
            # Categorical classification problem (y_pred_raw has shape (batch_size, num_classes))
            loss = tf.keras.losses.categorical_crossentropy(y_true_raw, y_pred_raw)
        mean_loss = tf.reduce_mean(loss).numpy()

        return mean_loss
