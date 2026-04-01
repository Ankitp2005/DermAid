import torch
import torch.nn as nn
import torch.nn.functional as F

class DermAidLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=1.0, beta=0.5, gamma=0.1):
        super(DermAidLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.condition_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.severity_loss = nn.CrossEntropyLoss()
        
        # Binary Cross Entropy for confidence prediction (which outputs via Sigmoid)
        self.conf_loss = nn.BCELoss()

    def forward(self, cond_pred, sev_pred, conf_pred, cond_true, sev_true):
        # Primary condition classification loss
        l_cond = self.condition_loss(cond_pred, cond_true)
        
        # Severity outcome classification loss
        l_sev = self.severity_loss(sev_pred, sev_true)
        
        # Confidence penalty: network should predict lower confidence if it gets the condition wrong
        # The target for the confidence head is "was the argmax correct?"
        with torch.no_grad():
            cond_correct = (torch.argmax(cond_pred, dim=1) == cond_true).float()
            
        # Ensure confidence predictions match the batch shape of cond_correct
        l_conf = self.conf_loss(conf_pred.squeeze(), cond_correct)
        
        # Total combined loss
        total = (self.alpha * l_cond) + (self.beta * l_sev) + (self.gamma * l_conf)
        
        return total, l_cond, l_sev, l_conf


def focal_loss(inputs, targets, gamma=2.0, alpha=None):
    """
    Standalone focal loss implementation. 
    Can be used as a drop-in replacement for CrossEntropyLoss on highly imbalanced, hard datasets.
    
    Args:
        inputs: Predictions (logits) of shape [N, C]
        targets: Ground truth class indices of shape [N]
        gamma: Focusing parameter, typically >= 0
        alpha: Optional class weights tensor of shape [C]
    """
    # Cross Entropy Loss without reduction returns the loss for each item
    ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)
    
    # pt is the probability of the true class
    pt = torch.exp(-ce_loss)
    
    # Calculate focal loss element-wise
    f_loss = ((1 - pt) ** gamma) * ce_loss
    
    # Return the mean loss across the batch
    return torch.mean(f_loss)
