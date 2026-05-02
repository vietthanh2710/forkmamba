import torch
import torch.nn as nn
import numpy as np

def iou_score(y_pred, y_true, smooth = 1e-5):
    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred & y_true).sum()
    union = (y_pred | y_true).sum()

    return (intersection + smooth) / (union + smooth)

def dice_score(y_pred, y_true, smooth=1e-5):
    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred * y_true).sum()

    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def recall_score(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred > threshold
    y_true = y_true == torch.max(y_true)
    TP = ((y_pred==1)&(y_true==1))
    FN = ((y_pred==0)&(y_true==1))
    Recall = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FN)) + 1e-5)
    return Recall

def y_predecision_score(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred > threshold
    y_true = y_true == torch.max(y_true)
    TP = ((y_pred==1)&(y_true==1))
    FP = ((y_pred==1)&(y_true==0))
    Precision = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FP)) + 1e-5)
    return Precision