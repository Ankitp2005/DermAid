import numpy as np
import torch

def mixup_data(x, y, alpha=0.4, device='cuda'):
    """
    Applies MixUp augmentation to a batch of images.
    
    MixUp creates convex combinations of pairs of examples and their labels.
    This is critical for dermoscopy datasets as it acts as a strong regularizer,
    smoothing the decision boundary between visually similar classes like nv 
    (nevus) and mel (melanoma). It reduces overconfidence and improves generalization.
    
    Args:
        x (torch.Tensor): Output tensor of images.
        y (torch.Tensor): Labels.
        alpha (float): Beta distribution parameter.
        device (str): Device to place the tensors on.
        
    Returns:
        mixed_x, y_a, y_b, lam: The mixed images, original labels, shuffled labels, and lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the loss using the original labels, the shuffled labels, and lambda.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix(x, y, alpha=1.0, device='cuda'):
    """
    Applies CutMix augmentation to a batch of images.
    
    CutMix replaces a random patch of an image with a patch from another image.
    Like MixUp, this is critical because it smooths the decision boundary between 
    classes (e.g., nv vs mel) by forcing the model to rely on multiple local features 
    rather than a single discriminative region.
    
    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Labels.
        alpha (float): Beta distribution parameter.
        device (str): Device to place the tensors on.
        
    Returns:
        mixed_x, y_a, y_b, lam: The mixed images, original labels, shuffled labels, and the true lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match the ratio of the mixed area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return mixed_x, y_a, y_b, lam
