from metrics.Metric import Metric
import torch
import torch.nn.functional as F


class LossPrivileged(Metric):

    def __init__(self):
        name = "LossPrivileged"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):
        mask_s1 = torch.where(s == 1)[0]  # Extract indices

        # Select only the relevant samples
        y_true_s1 = y_true_raw[mask_s1]
        y_pred_s1 = y_pred_raw[mask_s1]

        if y_pred_raw.shape[1] == 1:
            # Binary classification
            y_true_s1 = y_true_s1.view(-1)  # Flatten
            y_pred_s1 = y_pred_s1.view(-1)  # Flatten
            loss_s1 = F.binary_cross_entropy(y_pred_s1, y_true_s1)
        else:
            # Categorical classification
            loss_s1 = F.cross_entropy(y_pred_s1, y_true_s1)

        return loss_s1.mean().item()
