from metrics.Metric import Metric
import torch
import torch.nn.functional as F


class Loss(Metric):

    def __init__(self):
        name = "Loss"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):
        if y_pred_raw.shape[1] == 1:
            # Binary classification (y_pred_raw has shape (batch_size, 1))
            y_pred_raw_reshaped = torch.squeeze(y_pred_raw, dim=1)  # Remove single dimension
            loss = F.binary_cross_entropy(y_pred_raw_reshaped, y_true_raw)
        else:
            # Categorical classification (y_pred_raw has shape (batch_size, num_classes))
            loss = F.cross_entropy(y_pred_raw, y_true_raw)

        return loss.mean().item()
