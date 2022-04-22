import torch
import torch.nn as nn
from . import metrics


class JaccardLoss(nn.Module):
    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "JaccardLoss"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        print(input.size(), target.size())
        for i in range(0, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.iou(ypr, ygt)
        return losses


class DiceLoss(nn.Module):
    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "DiceLoss"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fcore(ypr, ygt)
        return losses

class CEWithLogitsLoss(nn.Module):
    def __init__(self, weights,device="cuda"):
        super().__init__()
        self.weight = torch.from_numpy(weights).float().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        self.name = "CELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        loss = loss.mean((1,2))
        return loss

class CEWithLogitsLoss_(nn.Module):
    def __init__(self, weights, device="cuda"):
        super().__init__()
        self.weight = torch.from_numpy(weights).float().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.name = "CELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss
