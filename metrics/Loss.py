from metrics.Metric import Metric

import tensorflow as tf
import torch
import torch.nn.functional as F

class Loss(Metric):

    def __init__(self):
        name = "Loss"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):

        print(y_true_raw)
        print(y_pred_raw)


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
        print(mean_loss)
        y_true_raw = torch.tensor(y_true_raw, dtype=torch.float32)
        y_pred_raw = torch.tensor(y_pred_raw, dtype=torch.float32)
        loss_py_torch = F.cross_entropy(y_pred_raw, y_true_raw)
        mean_loss = loss_py_torch.item()
        print(mean_loss)
        exit()

        return mean_loss
