import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import wandb

imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

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

def visualize_rescale_image(image, tag):
    # features : B x C x H x W
    if image.size()[0] > 1:
        for batch_idx in range(3):
            img = image[batch_idx].detach().cpu()
            img = img.numpy()
            img = 255 * (img * imagenet_std + imagenet_mean)
            img = img.astype(np.uint8)
            # img = img.numpy().squeeze()
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)

            wandb.log({f"{tag}_{batch_idx}" : [wandb.Image(img)]})

def visualize_cam(image, cam, tag):
    colormap = cv2.COLORMAP_JET 
    image_origin = image.detach().cpu()
    cam_origin = cam.detach().cpu()
    
    for batch_idx in range(3):
        img = image_origin[batch_idx]
        img = img.numpy()
        img = 255 * (img * imagenet_std + imagenet_mean)
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)

        cam = cam_origin[batch_idx] # (h, w)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image_origin.size()[2:], mode='bilinear', align_corners=True)
        cam = cam.squeeze().unsqueeze(0).numpy()
        cam = (cam - np.min(cam)) / np.ptp(cam)
        cam = ((1 - cam) * 255).astype(np.uint8)
        cam = np.transpose(cam, (1, 2, 0)) # (h, w, 1)
        cam = cv2.applyColorMap(cam, colormap)
        cam = Image.fromarray(cam)
        cam.putalpha(127)

        img.paste(cam, (0, 0), cam)

        wandb.log({f"{tag}_{batch_idx}" : [wandb.Image(img)]}, commit=False)

