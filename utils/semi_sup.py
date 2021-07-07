import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils.losses import *
import wandb, random
from copy import deepcopy

def semi_sup_learning(self, input_ul, label_ul):
    input_ul = self.upsampler_ul(input_ul)
    batch = input_ul.shape[0]
    student_label = self.model(input_ul)

    with torch.no_grad():
        teacher_label = self.teacher(input_ul).detach()

    vat_loss = VAT(model = self.model, n_power=1, XI=1e-6, eps=3)
    lds = vat_loss(input_ul, student_label)

    return lds

def adv_self_training(self, input_l, teacher_output, label_l, input_ul):
    input_ul = self.upsampler_ul(input_ul)
    self.optimizer2.zero_grad()

    teacher_soft_label = torch.softmax(teacher_output, dim=1)

    student_output = self.sec_student(input_l)

    loss = []
    softloss_sup = softmax_kl_loss(student_output, teacher_soft_label) / student_output.shape[0]
    hardloss_sup = CEloss(student_output, label_l)

    teacher_output_unsup = self.model(input_ul)

    student_output_unsup = self.sec_student(input_ul)
    teacher_soft_label_unsup = torch.softmax(teacher_output_unsup, dim=1)
    teacher_hard_label_unsup = torch.argmax(teacher_output_unsup, dim=1)

    softloss_sup_unsup  = softmax_kl_loss(student_output_unsup, teacher_soft_label_unsup) / teacher_output_unsup.shape[0]
    hardloss_sup_unsup  = CEloss(student_output_unsup, teacher_hard_label_unsup)

    loss.append(softloss_sup)
    loss.append(hardloss_sup)

    loss.append(softloss_sup_unsup)
    loss.append(hardloss_sup_unsup)

    t_loss = total_loss(loss)

    self.scaler.scale(t_loss).backward(retain_graph=True)
    self.scaler.step(self.optimizer2)
    self.scaler.update()

    wandb.log({"self_train/softloss_sup" : softloss_sup})
    wandb.log({"self_train/hardloss_sup" : hardloss_sup})
    wandb.log({"self_train/softloss_sup_unsup" : softloss_sup_unsup})
    wandb.log({"self_train/hardloss_sup_unsup" : hardloss_sup_unsup})
    