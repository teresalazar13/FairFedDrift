from metrics.Metric import Metric

import tensorflow as tf


class LossUnprivileged(Metric):

    def __init__(self):
        name = "LossUnprivileged"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):
        mask_s0 = tf.where(s == 0)
        y_true_s0 = tf.gather(y_true_raw, mask_s0)
        y_pred_s0 = tf.gather(y_pred_raw, mask_s0)

        if y_pred_raw.shape[1] == 1:
            # Binary classification problem (y_pred_raw has shape (batch_size, 1))
            y_true_s0_reshaped = tf.reshape(y_true_s0, [len(y_true_s0)])
            y_pred_s0_reshaped = tf.reshape(y_pred_s0, [len(y_pred_s0)])
            #logging.info(tf.shape(y_true_s0_reshaped))
            #logging.info(tf.shape(y_pred_s0_reshaped))
            loss_s0 = tf.keras.losses.binary_crossentropy(y_true_s0_reshaped, y_pred_s0_reshaped)
        else:
            # Categorical classification problem (y_pred_raw has shape (batch_size, num_classes))
            #logging.info(tf.shape(y_true_s0))
            #logging.info(tf.shape(y_pred_s0))
            loss_s0 = tf.keras.losses.categorical_crossentropy(y_true_s0, y_pred_s0)

        mean_loss_s0 = tf.reduce_mean(loss_s0).numpy()

        return mean_loss_s0