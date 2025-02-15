from metrics.Metric import Metric
import torch
import torch.nn.functional as F


class LossUnprivileged(Metric):

    def __init__(self):
        name = "LossUnprivileged"
        super().__init__(name)

    def calculate(self, _, __, y_true_raw, y_pred_raw, s):
        mask_s0 = torch.where(s == 0)[0]  # Extract indices

        # Select only the relevant samples
        y_true_s0 = y_true_raw[mask_s0]
        y_pred_s0 = y_pred_raw[mask_s0]

        if y_pred_raw.shape[1] == 1:
            # Binary classification
            y_true_s0 = y_true_s0.view(-1)  # Flatten
            y_pred_s0 = y_pred_s0.view(-1)  # Flatten
            loss_s0 = F.binary_cross_entropy(y_pred_s0, y_true_s0)
        else:
            # Categorical classification
            loss_s0 = F.cross_entropy(y_pred_s0, y_true_s0)

        return loss_s0.mean().item()
