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

def visualize_rescale_image(mean, std, image, tag): # vis image itself with mean train
    # features : B x C x H x W
    origin_image = image
    for batch_idx in range(image.shape[0]):
        image = origin_image[batch_idx].detach().cpu()
        X = un_normalize(image, mean, std)
        X = image.numpy().squeeze()

        # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
        original_image = (255*(X - np.min(X))/np.ptp(X)).astype(np.uint8)

        #print("original image shape : ", original_image.shape)
        wandb.log({str(tag)+"_"+str(batch_idx) : [wandb.Image(np.transpose(X, (1,2,0)))]})

def visualize_cam(image, cam, mean, std, tag):
    cam_origin = cam
    colormap: int = cv2.COLORMAP_JET
    image_origin = image.detach().cpu()
    
    for batch_idx in range(image.shape[0]):
        image = image_origin[batch_idx]
        X = un_normalize(image, mean, std).numpy()
        img = (X-np.min(X))/(np.max(X)-np.min(X))
        img = np.transpose(img, (1,2,0))
        
        cam = cam_origin.cpu().detach()
        cam = cam[batch_idx] # h x w
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image.size()[1:], mode='bilinear', align_corners=True)
        cam = cam.squeeze().unsqueeze(0).numpy()
        cam = (255*(cam - np.min(cam))/np.ptp(cam)).astype(np.uint8)
        cam = np.transpose(cam, (1,2,0)) # h x w x 1
        
        heatmap = cv2.applyColorMap(cam, colormap)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap * 0.3 + img * 0.7
        #cam = (cam / np.max(cam)
        cam = (255*(cam - np.min(cam))/np.ptp(cam)).astype(np.uint8)
        
        wandb.log({str(tag+"_"+str(batch_idx)) : [wandb.Image(cam)]}, commit=False)
