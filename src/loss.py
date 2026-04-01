import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class DermAidLoss(nn.Module):
    """
    Multi-task loss function for DermAid model:
    1. Condition classification loss (weighted CrossEntropy)
    2. Severity classification loss (CrossEntropy)
    3. Confidence regularization loss (BCE)
    """

    def __init__(self, class_weights, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        # alpha: weight for condition loss (primary)
        # beta:  weight for severity loss
        # gamma: weight for confidence regularization
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.condition_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.severity_loss = nn.CrossEntropyLoss()
        self.conf_loss = nn.BCELoss()

    def forward(self, cond_pred, sev_pred, conf_pred, cond_true, sev_true):
        l_cond = self.condition_loss(cond_pred, cond_true)
        l_sev = self.severity_loss(sev_pred, sev_true)

        # Confidence regularization: model should be confident when correct
        cond_correct = (torch.argmax(cond_pred, dim=1) == cond_true).float()
        l_conf = self.conf_loss(conf_pred.squeeze(), cond_correct)

        total = self.alpha * l_cond + self.beta * l_sev + self.gamma * l_conf
        return total, l_cond, l_sev, l_conf
