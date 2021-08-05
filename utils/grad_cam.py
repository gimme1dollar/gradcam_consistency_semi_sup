import torch
import torch.multiprocessing
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from efficientnet_pytorch.model import EfficientNet

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T
from torchvision import datasets
from torchvision.datasets import ImageFolder

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

class ActivationsAndGradients:
    """ Class for extracting activations and gradients 
    from targetted intermediate layers.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = torch.squeeze(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = torch.squeeze(grad_output[0])
    
    def clear_list(self):
        del self.gradients
        torch.cuda.empty_cache()
        self.gradients = None
    
    def buffer_clear(self):
        del self.gradients, self.activations
        torch.cuda.empty_cache()
        self.gradients = None
        self.activations = None
        
    def __call__(self, x):
        self.gradients = None
        self.activations = None 
        return self.model(x)


class GradCAM:
    """ Class for extracting GradCAM from target model.
    Args:
      model : target model being examined
      target_layer : intermediate layer being examined
      input_tensor : input image with size of (b, c, h, w)
      expand_size : the size of output GradCAM (e.g., (224,224))
    Return:
      Set of class activation maps, with length of batch size
    """
    def __init__(self, model, target_layer):
        self.model = model.to(device)
        self.target_layer = target_layer
        self.activations_and_grads = ActivationsAndGradients(model, target_layer)
            
    def buffer_clear(self):
        self.activations_and_grads.buffer_clear()

    def __call__(self, input_tensor, expand_size):
        self.buffer_clear()
        self.model.eval()
        
        cam_stack=[]    
        for batch_idx in range(input_tensor.shape[0]): # iteration w.r.t. batch
            self.model.zero_grad()
            img = input_tensor[batch_idx]
            img = img.unsqueeze(0) # (c, h, w) -> (b, c, h, w)
            output = self.activations_and_grads(img)[0] 

            y_c = output[torch.argmax(output)] # GAP over channel
            y_c.backward(retain_graph=True)

            activations = self.activations_and_grads.activations
            grads = self.activations_and_grads.gradients
            weights = torch.mean(grads, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations, dim=0)

            cam = cam.unsqueeze(0).unsqueeze(0) # (h, w) -> (b, c, h, w)
            cam = F.interpolate(cam, size=expand_size, mode='bilinear', align_corners=True)

            # Normalize
            min_v = torch.min(cam)
            range_v = torch.max(cam) - min_v
            if range_v > 0:
                cam = (cam - min_v) / range_v
            else:
                cam = torch.zeros(cam.size())
            cam_stack.append(cam.to(device))
            
            self.activations_and_grads.clear_list()
            del y_c, activations, grads, weights, cam, output

        concated_cam = torch.cat(cam_stack, dim=0).squeeze()
        del cam_stack, input_tensor
        torch.cuda.empty_cache()
        self.model.train()
        self.buffer_clear()
        return concated_cam