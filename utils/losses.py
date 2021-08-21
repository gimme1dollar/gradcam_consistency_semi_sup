# Implemented upon pytorch 1.2.0
import torch.nn as nn
import torchvision.transforms.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

def constrative_loss(div1_origin, div2_new, target):
    cons_loss = ContrastiveLoss()(div1_origin, div2_new, target.cuda())
    return cons_loss

def CEloss(inputs, gt):
    return nn.CrossEntropyLoss()(inputs, gt)

def MSEloss(inputs, gt):
    c = nn.MSELoss()
    loss = c(inputs,gt)
    return loss

def total_loss(losses_list):
    total = 0
    for component in losses_list:
        if isinstance(component, list):
            total += sum(component)
        else:
            total += component
    return total

