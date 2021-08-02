from typing import Union, Optional, List, Tuple, Text, BinaryIO
from torchvision.utils import make_grid
import numpy as np
import cv2
import torch
from PIL import Image
import torch.nn.functional as F
import wandb

def renormalize_float(vector, range_t : tuple):

    row = torch.Tensor(vector)
    r = torch.max(row) - torch.min(row)
    row_0_to_1 = (row - torch.min(row)) / r
    r2 = range_t[1] - range_t[0]
    row_normed = (row_0_to_1 * r2) + range_t[0]

    return row_normed.numpy()

def un_normalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def visualize_rescale_image(image, tag): # vis image itself with mean train
    # features : B x C x H x W
    for batch_idx in range(3):
        img = image[batch_idx].detach().cpu()
        img = img.numpy().squeeze()
        img = np.transpose(img, (1,2,0))

        wandb.log({str(tag)+"_"+str(batch_idx) : [wandb.Image(img)]})

def visualize_cam(image, cam, tag):
    colormap: int = cv2.COLORMAP_JET
    image_origin = image.detach().cpu()
    cam_origin = cam.detach().cpu()
    
    for batch_idx in range(3):
        img = image_origin[batch_idx]
        img = np.transpose(img.numpy(), (1,2,0))
        
        cam = cam_origin[batch_idx] # (h, w)
        
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image_origin.size()[2:], mode='bilinear', align_corners=True)
        cam = cam.squeeze().unsqueeze(0).numpy()
        cam = (255*(cam - np.min(cam))/np.ptp(cam)).astype(np.uint8)
        cam = np.transpose(cam, (1,2,0)) # (h, w, 1)
        
        heatmap = cv2.applyColorMap(cam, colormap)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap * 0.9 + img * 0.1
        cam = (255*(cam - np.min(cam))/np.ptp(cam)).astype(np.uint8)
        
        wandb.log({str(tag+"_"+str(batch_idx)) : [wandb.Image(cam)]}, commit=False)
