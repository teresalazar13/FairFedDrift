from metrics.Metric import Metric

import tensorflow as tf


class LossPrivileged(Metric):

    def __init__(self):
        name = "LossPrivileged"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):
        mask_s1 = tf.where(s == 1)
        y_true_s1 = tf.gather(y_true_raw, mask_s1)
        y_pred_s1 = tf.gather(y_pred_raw, mask_s1)

        if y_pred_raw.shape[1] == 1:
            # Binary classification problem (y_pred_raw has shape (batch_size, 1))
            y_true_s1_reshaped = tf.reshape(y_true_s1, [len(y_true_s1)])
            y_pred_s1_reshaped = tf.reshape(y_pred_s1, [len(y_pred_s1)])
            #print(tf.shape(y_true_s1_reshaped))
            #print(tf.shape(y_pred_s1_reshaped))
            loss_s1 = tf.keras.losses.binary_crossentropy(y_true_s1_reshaped, y_pred_s1_reshaped)
        else:
            # Categorical classification problem (y_pred_raw has shape (batch_size, num_classes))
            loss_s1 = tf.keras.losses.categorical_crossentropy(y_true_s1, y_pred_s1)

        mean_loss_s1 = tf.reduce_mean(loss_s1).numpy()

        return mean_loss_s1
