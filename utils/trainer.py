import torch
import torch.multiprocessing
import torchvision.transforms.functional as VF
from efficientnet_pytorch.model import EfficientNet

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T
from torchvision import datasets
from torchvision.datasets import ImageFolder
from typing import Callable

import math
import warnings
import random
import numpy as np
import os
import glob
import os.path, datetime, time
import matplotlib.pyplot as plt
from os.path import join as pjn
from PIL import Image
from typing import Union

import wandb, argparse
from tqdm import tqdm

from utils.losses import *
from utils.visualize import *
from utils.grad_cam import *
from dataset.dataloader import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
    
class TrainManager(object):
    def __init__(
            self,
            model,
            optimizer,
            args,
            additional_cfg,
            label_loader,
            unlabel_loader,
            val_loader,
            scaler=None,
            num_classes=None,
            ):
        self.model = model
        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader
        self.optimizer = optimizer
        self.args = args
        self.add_cfg = additional_cfg
        self.tbx_wrtr_dir = additional_cfg.get('tbx_wrtr_dir')
        self.scaler = scaler
        self.val_loader = val_loader
        self.num_classes = num_classes

        # Network target layer
        if args.exp_net == "EfficientNet":
            for idx, module in model._modules.items():
                for name, block in enumerate(getattr(model, "_blocks")):
                    if str(name) == '5' and self.args.exp_layer == 1:
                        target_layer = block
                    elif str(name) == '10' and self.args.exp_layer == 2:
                        target_layer = block
                    elif str(name) == '20' and self.args.exp_layer == 3:
                        target_layer = block
                    elif str(name) == '30' and self.args.exp_layer == 4:
                        target_layer = block
        elif args.exp_net == "ResNet":
            if self.args.exp_layer == 1:
                target_layer = self.model.layer1[-1]
            elif self.args.exp_layer == 2:
                target_layer = self.model.layer2[-1]
            elif self.args.exp_layer == 3:
                target_layer = self.model.layer3[-1]
            elif self.args.exp_layer == 4:
                target_layer = self.model.layer4[-1]
        
        self.get_cam = GradCAM(model=self.model, target_layer=target_layer)

        # Image size
        if args.exp_data == "dl20":
            factor = 4 #2
        elif args.exp_data == "cifar10":
            factor = 8 #4
        self.upsampler = torch.nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        
        self.resize_transform = T.Compose([
            T.Resize((256, 256))
        ])


    def save_outputs_hook(self) -> Callable:
        def fn(_, __, output):
            #print(output.size())
            self.save_feat.append(output)
        return fn

    def save_grad_hook(self) -> Callable:
        def fn(grad):
            self.save_grad.append(grad)
        return fn

    def color_wk_augmentation(self, img):
        color_transform = T.Compose([
            T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        ])

        return color_transform(img)

    def color_st_augmentation(self, img):
        color_transform = T.Compose([
            T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        ])

        return color_transform(img)

    def get_crop_params(self, img):
        w_, h_ = img.size(2), img.size(3)
        xl = random.randint(0, w_ / 4)
        xr = 0
        while (((xr - xl) < (w_ * 3 / 4)) or (xr <= xl)):
            xr = random.randint(xl, w_)

        yl = random.randint(0, h_ / 4)
        yr = 0
        while (((yr - yl) < (h_ * 3 / 4)) or (yr <= yl)):
            yr = random.randint(yl, h_)

        return xl, yl, xr-xl, yr-yl

    def get_rotate_params(self):
        choice = random.choice([0, 90, 180, 360])
        return choice

    def validate(self, model, device, topk=(1,3,5)):
        model.eval()
        total = 0
        maxk = max(topk)
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0

        with torch.no_grad():
            for b_idx, (image, labels) in tqdm(enumerate(self.val_loader), desc="validation", leave=False):
                image = self.upsampler(image)
                image = image.to(device)
                labels = labels.to(device)

                total += image.shape[0]
                
                outputs = model(image) # b x 1

                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = (pred == labels.unsqueeze(dim=0)).expand_as(pred)

                for k in topk:
                    if k == 1:
                        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
                        correct_1 += correct_k.item()
                    elif k == 3:
                        correct_k = correct[:3].reshape(-1).float().sum(0, keepdim=True)
                        correct_3 += correct_k.item()
                    elif k == 5:
                        correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
                        correct_5 += correct_k.item()
                    else:
                        raise NotImplementedError("Invalid top-k num")
        return (correct_1 / total) * 100, (correct_3 / total) * 100, (correct_5 / total) * 100

    def train(self, verbose=False):
        start = time.time()
        iter_per_epoch = len(self.label_loader)   
        print("  experiment name: ", self.args.exp_mode)
        print("  experiment mode: ", self.args.exp_mode)
        print("  experiment network: ", self.args.exp_net)
        print("  experiment layer: ", self.args.exp_layer)
        print("  experiment dataset: ", self.args.exp_data)
        print("  experiment ratio: ", self.args.ratio)
        print("  -------------------------- ")
        print("  batch size for training: ", self.args.batch_size_train)
        print("  batch size for validation: ", self.args.batch_size_val)
        print("  batch size for testing: ", self.args.batch_size_test)
        print("  -------------------------- ")
        print("  iteration per epoch(considered batch size): ", iter_per_epoch)
        print("  label iter : ", len(self.label_loader))
        print("  unlabel iter : ", len(self.unlabel_loader))
        print("  val iter : ", len(self.val_loader))
        print("  Progress bar for training epochs:")
        end_epoch = self.args.start_epoch + self.args.num_epochs

        best = -float("inf")
        unlabel_dataloader = iter(cycle(self.unlabel_loader))
        p_cutoff = 0.85
        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):
            self.model.train()

            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            for t_idx, (image, target) in tqdm(enumerate(self.label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                ### Labeled data
                image = self.upsampler(image)
                image = image.to(device) 
                target = target.to(device)

                self.optimizer.zero_grad()
                losses_list = []
                with torch.cuda.amp.autocast():
                    outputs = self.model(image)
                    sup_loss = CEloss(outputs, target)
                    losses_list.append(sup_loss)   
                wandb.log({"training/sup_loss" : sup_loss})

                if t_idx % 10 == 0:
                    visualize_rescale_image(image, "image_sup/image")

                ### Unlabeled dataset
                if self.args.exp_mode == "abla":
                    image_ul = next(unlabel_dataloader)
                    images = []
                    for i in image_ul[0]:
                        images.append(i)
                    image_ul = torch.stack(images)
                    image_ul = image_ul.to(device)
                    image_ul = self.upsampler(image_ul)
                    
                    ## Augmentation
                    wk_image = self.color_wk_augmentation(image_ul)
                    wk_image = wk_image.to(device)
                    
                    wk_label = self.model(wk_image)
                    wk_prob, wk_pred = torch.max(wk_label, dim=-1)
                    mask_p = wk_prob.ge(p_cutoff).float()

                    st_image = self.color_st_augmentation(image_ul)
                    i, j, h, w = self.get_crop_params(st_image)
                    cr_image = VF.crop(st_image, i, j, h, w)
                    cr_image = self.resize_transform(cr_image)
                    r = self.get_rotate_params()
                    rt_image = VF.rotate(cr_image, r)
                    rt_image = rt_image.to(device)
                    st_label = self.model(rt_image)
                    
                    if t_idx % 10 == 0:
                        visualize_rescale_image(image_ul, "image_org/image")
                        #visualize_rescale_image(wk_image, "image_wk_aug/image")
                        #visualize_rescale_image(st_image, "image_st_aug/image")
                        #visualize_rescale_image(cr_image, "image_cr_aug/image")
                        visualize_rescale_image(rt_image, "image_rt_aug/image")
                        
                    ## Loss
                    wk_pred_, st_label_ = [], []
                    for idx, mask in enumerate(mask_p):
                        if mask == 1:
                            wk_pred_.append(wk_pred[idx])
                            st_label_.append(st_label[idx])

                    if len(wk_pred_) > 0:
                        wk_pred_ = torch.stack(wk_pred_)
                        st_label_ = torch.stack(st_label_)
                                
                        self.optimizer.zero_grad()
                        with torch.cuda.amp.autocast():
                            label_loss = CEloss((st_label_), (wk_pred_).long())
                            label_loss *= self.args.alpha
                            losses_list.append(label_loss)   
                            wandb.log({"training/lbl_loss" : label_loss})
                     
                elif self.args.exp_mode == "grad":
                    image_ul = next(unlabel_dataloader)
                    images = []
                    for i in image_ul[0]:
                        images.append(i)
                    image_ul = torch.stack(images)
                    image_ul = image_ul.to(device)
                    image_ul = self.upsampler(image_ul)
                           
                    ## Augmentation
                    wk_image = self.color_wk_augmentation(image_ul)
                    wk_image = wk_image.to(device)
                    
                    wk_label = self.model(wk_image)
                    wk_prob, wk_pred = torch.max(wk_label, dim=-1)
                    mask_p = wk_prob.ge(p_cutoff).float()

                    st_image = self.color_st_augmentation(image_ul)
                    i, j, h, w = self.get_crop_params(st_image)
                    cr_image = VF.crop(st_image, i, j, h, w)
                    cr_image = self.resize_transform(cr_image)
                    r = self.get_rotate_params()
                    rt_image = VF.rotate(cr_image, r)
                    rt_image = rt_image.to(device)

                    st_label = self.model(rt_image)

                    ## Getting cam
                    wk_cam = self.get_cam(wk_image, image.size()[2:])
                    st_cam = self.get_cam(rt_image, image.size()[2:])
                    gt_cam = VF.crop(wk_cam, i, j, h, w)
                    gt_cam = self.resize_transform(gt_cam)
                    gt_cam = VF.rotate(gt_cam, r)

                    if verbose:
                        print(image.size())
                        print(image_ul.size())
                        print(wk_image.size())
                        print(cr_image.size())
                        print(wk_cam.size())
                        print(st_cam.size())
                        print(gt_cam.size())
                    
                    if t_idx % 10 == 0:
                        visualize_rescale_image(image_ul, "image_org/image")
                        visualize_rescale_image(rt_image, "image_rt_aug/image")

                        visualize_cam(wk_image, wk_cam, "wk_cam/cam")  
                        visualize_cam(rt_image, st_cam, "st_cam/cam") 
                        visualize_cam(rt_image, gt_cam, "gt_cam/cam")     
                    del wk_cam
                    
                    ## Loss
                    wk_pred_, st_label_, st_cam_, gt_cam_ = [], [], [], []
                    for idx, mask in enumerate(mask_p):
                        if mask == 1:
                            wk_pred_.append(wk_pred[idx])
                            st_label_.append(st_label[idx])
                            st_cam_.append(st_cam[idx])
                            gt_cam_.append(gt_cam[idx])

                    if len(wk_pred_) > 0:
                        wk_pred_ = torch.stack(wk_pred_)
                        st_label_ = torch.stack(st_label_)
                        st_cam_ = torch.stack(st_cam_)
                        gt_cam_ = torch.stack(gt_cam_)
                                
                        self.optimizer.zero_grad()
                        with torch.cuda.amp.autocast():
                            label_loss = CEloss((st_label_), (wk_pred_).long())
                            label_loss *= self.args.alpha
                            losses_list.append(label_loss)   
                            wandb.log({"training/lbl_loss" : label_loss})
                            
                            cam_loss = MSEloss(st_cam_, gt_cam_)
                            cam_loss *= self.args.beta
                            if not math.isnan(cam_loss): 
                                losses_list.append(cam_loss)
                            wandb.log({"training/cam_loss" : cam_loss})
                            
                    del st_cam, gt_cam

                ## Train model
                self.model.train()
                t_loss = total_loss(losses_list)
                wandb.log({"training/tloss" : t_loss})

                t_loss = total_loss(losses_list)
                self.scaler.scale(t_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()


            if epoch % 3 == 1:
                top1_acc, top3_acc, top5_acc = self.validate(self.model, self.add_cfg['device'])
                wandb.log({"validation/top1_acc" : top1_acc, "validation/top3_acc" : top3_acc, "validation/top5_acc" : top5_acc})

                if best < top1_acc:
                    best = top1_acc
                    self.save_ckpt()
            wandb.log({"validation/best_top1" : best})
            p_cutoff = min(p_cutoff + (0.1 / self.args.num_epochs), 1.0)
            
        end = time.time()   
        print("Total training time : ", str(datetime.timedelta(seconds=(int(end)-int(start)))))
        print("Finish.")

    def save_ckpt(self):
        nm = f'best_model.pth'

        if not os.path.isdir(pjn('checkpoints', self.tbx_wrtr_dir)):
            os.mkdir(pjn('checkpoints', self.tbx_wrtr_dir))

        fpath=pjn('checkpoints', self.tbx_wrtr_dir, nm)

        d = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(d, fpath)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

