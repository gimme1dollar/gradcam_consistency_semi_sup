# Implemented upon pytorch 1.2.0
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
import random

def CEloss(inputs, gt, ignore_index=255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(inputs, gt)

def total_loss(losses_list):
    total = 0
    for component in losses_list:
        if isinstance(component, list):
            total += sum(component)
        else:
            total += component
    return total

def cosine_loss(x1, x2, y):
    
    p_loss = 0
    p_cnt = 1e-12
    n_loss = 0
    n_cnt = 1e-12

    for idx in range(x1.shape[0]):
        cos = torch.cosine_similarity(x1[idx], x2[idx], dim=0)
        if y[idx] == 1:
            p_loss += 3 - cos
            p_cnt += 1
            wandb.log({"training/same_sim" : cos}, commit=False)
            wandb.log({"training/same_sim_1-cos" : 3 - cos}, commit=False)

        elif y[idx] == -1:
            n_loss += max(0, cos + 0.5)
            n_cnt += 1
            wandb.log({"training/diff_sim" : max(0, cos + 0.5)}, commit=False)

    pm = p_loss / p_cnt
    nm = n_loss / n_cnt
    
    wandb.log({"training/same_avg_loss" : pm}, commit=False)
    wandb.log({"training/diff_avg_loss" : nm}, commit=False)
    return pm, nm

def triplet_loss(image, target, anc_vector):

    tp_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    loss = 0

    for oidx in range(image.shape[0]):
        pos_list = []
        neg_list = []
        for iidx in range(image.shape[0]):
            label = (target[oidx] == target[iidx])
            if label.any(): # positive
                pos_list.append(iidx)
            else:
                neg_list.append(iidx)

        min_range = min(len(pos_list), len(neg_list))

        p1 = random.sample(pos_list, min_range)
        n1 = random.sample(neg_list, min_range)
        
        loss += tp_loss(anc_vector[oidx], anc_vector[p1], anc_vector[n1])

    loss = loss / (image.shape[0])
    wandb.log({"training/triplet_loss":loss})
    return loss


def softmax_kl_loss(input_logits, target_logits):
    input_log_softmax = F.log_softmax(input_logits, dim=0)
    target_softmax = F.softmax(target_logits, dim=0)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


# bring code from https://github.com/jik0730/VAT-pytorch
class VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model, n_power, XI, eps):
        super(VAT, self).__init__()
        self.model = model
        self.n_power = n_power
        self.XI = XI
        self.epsilon = eps

    def forward(self, X, logit):
        vat_loss = virtual_adversarial_loss(X, logit, self.model, self.n_power,
                                            self.XI, self.epsilon)
        return vat_loss  # already averaged


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1, 1)
    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI,
                                              epsilon):
    d = torch.randn_like(x)

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        logit_m = model(x + d)
        dist = kl_divergence_with_logit(logit, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, model, n_power, XI, epsilon):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model,
                                                       n_power, XI, epsilon)
    logit_p = logit.detach()
    logit_m = model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return loss