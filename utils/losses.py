import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        return 1 - (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

class WeightedConcaveRegionLoss(nn.Module):
    """
    A weighted loss function for region segmentation, inspired by the Dice Similarity Coefficient (DSC) score.
    This loss function dynamically adjusts its weighting of false positives and
    false negatives based on the training epoch, allowing it to focus on different
    aspects of the segmentation task at different stages of training.
    """
    def __init__(self, alpha, beta, b, c):
        """
        Initializes the loss function with hyperparameters.

        Args:
            alpha (float): Exponent for the false negative term, controls its magnitude.
            beta (float): Controls the slope of the sigmoid function for the false positive term.
            b (float): Mean of the Gaussian function, representing the epoch at which the
                       weighting is balanced between false positives and false negatives.
            c (float): Standard deviation of the Gaussian function, controlling the width of
                       the weighting curve. A smaller 'c' means a sharper, more focused
                       change in weighting around epoch 'b'.
        """
        super(WeightedConcaveRegionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.b = b  # Mean - center of the weight function (Gaussian)
        self.c = c  # Standard deviation - width of the weight function (Gaussian)

    def forward(self, y_pred, y_true, current_epoch):
        """
        Calculates the loss for a given batch of predictions.

        Args:
            y_pred (torch.Tensor): The raw model output (logits or scores).
            y_true (torch.Tensor): The ground truth segmentation mask (0 or 1).
            current_epoch (int): The current training epoch.

        Returns:
            torch.Tensor: The loss value.
        """
        y_pred = torch.sigmoid(y_pred)

        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        
        y_pred_pos = torch.clamp(y_pred_pos, min=1e-7, max=1.0)

        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos ** self.alpha))

        FP_y = 2 / (1 + torch.exp(-self.beta * y_pred_pos)) - 1
        false_pos = torch.sum((1 - y_true_pos) * FP_y)


        base = torch.tensor(- 1 * (current_epoch - self.b) ** 2 / (2 * self.c ** 2))
        delta = 0.5 * torch.exp(base) + 1
      
        smooth = 1e-3
        return (delta * false_neg + (2 - delta) * false_pos + smooth) / (2 * true_pos + smooth)
