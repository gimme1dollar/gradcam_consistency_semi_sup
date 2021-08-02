import torch
import torch.multiprocessing
import torchvision.transforms.functional as VF
from efficientnet_pytorch.model import EfficientNet

from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T
from torchvision import datasets
from dataset.dataloader import LoadDataset, LoadSemiDataset
from typing import Callable

import math
import warnings
import random
import numpy as np
import os, os.path, datetime, time
import matplotlib.pyplot as plt
from os.path import join as pjn
from utils.losses import *
from utils.visualize import *

import wandb, argparse
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

cifar10_mean = [0.4913, 0.4821, 0.4465]
cifar10_std = [0.2470, 0.2434, 0.2615]

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

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def init(val_batch_train, val_batch_valid, val_split, args):
    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug
    transform_cifar10 = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                T.Resize((64, 64))
            ])

    train_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=True, download=True, transform=transform_cifar10)
    valid_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=False, download=True, transform=transform_cifar10)
    
    split_ratio = val_split
    shuffle_dataset = True
    random_seed= 123

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    label_indices, unlabel_indices = indices[:split], indices[split:]

    label_sampler = SubsetRandomSampler(label_indices)
    unlabel_sampler = SubsetRandomSampler(unlabel_indices)
    
    label_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=val_batch_train, sampler=label_sampler)
    unlabel_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=val_batch_train, sampler=unlabel_sampler) 
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=val_batch_valid, shuffle=True)
    
    return label_loader, unlabel_loader, valid_loader
    
    
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

        for idx, module in model._modules.items():
            for name, block in enumerate(getattr(model, "_blocks")):
                if str(name) == '30': # may change the target layer
                    target_layer = block

        self.get_cam = GradCAM(model=self.model, target_layer=target_layer)

        self.to_tensor_transform = T.Compose([
            T.ToTensor()
        ])
        self.resize_transform = T.Compose([
            T.Resize((64, 64))
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
        color_transform = T.ColorJitter(brightness=(0.0, 0))
        res = color_transform(img)
        return img

    def color_st_augmentation(self, img):
        color_transform = T.ColorJitter(#brightness=(0.2, 2), 
                                    #contrast=(0.3, 2), 
                                    #saturation=(0.2, 2), 
                                    hue=(-0.3, 0.3))
        res = color_transform(img)
        return img

    def get_crop_params(self, img):
        w_, h_ = img.size(2), img.size(3)
        xl = random.randint(0, w_ / 8)
        xr = 0
        while (((xr - xl) < (w_ * 4 / 8)) or (xr <= xl)):
            xr = random.randint(xl, w_)

        yl = random.randint(0, h_ / 8)
        yr = 0
        while (((yr - yl) < (h_ * 4 / 8)) or (yr <= yl)):
            yr = random.randint(yl, h_)

        return xl, yl, xr, yr

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

    def train(self):
        start = time.time()
        epoch = 0
        iter_per_epoch = len(self.label_loader)
        print("  iteration per epoch(considered batch size): ", iter_per_epoch)
        print("  label iter : ", len(self.label_loader))
        print("  unlabel iter : ", len(self.unlabel_loader))
        print("  val iter : ", len(self.val_loader))
        print("  Progress bar for training epochs:")
        end_epoch = self.args.start_epoch + self.args.num_epochs

        unlabel_dataloader = iter(cycle(self.unlabel_loader))
        p_cutoff_init = 0.85
        p_cutoff_factor = 1.025
        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):
            p_cutoff = min(p_cutoff_init * p_cutoff_factor ** (epoch//5), 0.98)
            
            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            for t_idx, (image, target) in tqdm(enumerate(self.label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                
                ### Labeled data
                image = image.to(device) 
                target = target.to(device)

                self.optimizer.zero_grad()
                losses_list = []
                with torch.cuda.amp.autocast():
                    ### sup loss ###
                    outputs = self.model(image)
                    sup_loss = CEloss(outputs, target)
                    losses_list.append(sup_loss)   
                wandb.log({"training/sup_loss" : sup_loss})

                ### Unlabeled dataset
                image_ul = next(unlabel_dataloader)
                images = []
                for i in image_ul[0]:
                    images.append(i)
                image_ul = torch.stack(images)
                image_ul = image_ul.to(device)
                
                ## Augmentation
                wk_image = self.color_wk_augmentation(image_ul)
                wk_image = wk_image.cuda()
                
                wk_label = self.model(wk_image)
                wk_prob, wk_pred = torch.max(wk_label, dim=-1)
                mask_p = wk_prob.ge(p_cutoff).float()

                st_image = self.color_st_augmentation(image_ul)
                i, j, h, w = self.get_crop_params(st_image)
                cr_image = VF.crop(st_image, i, j, h, w)
                cr_image = self.resize_transform(cr_image)
                r = self.get_rotate_params()
                rt_image = VF.rotate(cr_image, r)
                rt_image = rt_image.cuda()

                st_label = self.model(rt_image)

                ## GradCAM
                wk_cam = self.get_cam(wk_image, image.size()[2:])
                st_cam = self.get_cam(rt_image, image.size()[2:])
                gt_cam = VF.crop(wk_cam, i, j, h, w)
                gt_cam = VF.rotate(gt_cam, r)
                gt_cam = self.resize_transform(gt_cam)

                ## Visualization
                if t_idx % 10 == 0:
                    visualize_rescale_image(image_ul, "image_org/image")
                    #visualize_rescale_image(wk_image, "image_wk_aug/image")
                    #visualize_rescale_image(st_image, "image_st_aug/image")
                    #visualize_rescale_image(cr_image, "image_cr_aug/image")
                    visualize_rescale_image(rt_image, "image_rt_aug/image")

                    visualize_cam(wk_image, wk_cam, "wk_cam/cam")  
                    visualize_cam(rt_image, st_cam, "st_cam/cam") 
                    visualize_cam(rt_image, gt_cam, "gt_cam/cam")    
                del wk_cam
                
                ## Loss
                mask_lbl = torch.stack([ torch.ones_like(st_label[0]) if int(mask_p[i]) else torch.zeros_like(st_label[0]) for i in range(gt_cam.size(0)) ])
                mask_cam = torch.stack([ torch.ones_like(gt_cam[0]) if int(mask_p[i]) else torch.zeros_like(gt_cam[0]) for i in range(gt_cam.size(0)) ])                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    label_loss = CEloss((st_label*mask_lbl), (wk_pred*mask_p).long())
                    label_loss *= args.alpha
                    losses_list.append(label_loss)   
                    wandb.log({"training/lbl_loss" : label_loss})
                    
                    cam_loss = MSEloss(st_cam * mask_cam, gt_cam * mask_cam)
                    cam_loss *= args.beta
                    if math.isnan(cam_loss) is False:
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

            if epoch % 30 == 1:
                self.save_ckpt(epoch)

            if epoch % 5 == 1:
                top1_acc, top3_acc, top5_acc = self.validate(self.model, self.add_cfg['device'])
                wandb.log({"validation/top1_acc" : top1_acc, "validation/top3_acc" : top3_acc, "validation/top5_acc" : top5_acc})
                top1_acc_stu = top1_acc

            self.adjust_learning_rate(epoch)
            
        end = time.time()   
        print("Total training time : ", str(datetime.timedelta(seconds=(int(end)-int(start)))))
        print("Finish.")

    def adjust_learning_rate(self, epoch):
        # update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            prev_lr = param_group['lr']
            param_group['lr'] = prev_lr * self.args.lr_anneal_rate

    def save_ckpt(self, epoch):
        if epoch % self.args.save_ckpt == 0:

            nm = f'epoch_{epoch:04d}.pth'

            if not os.path.isdir(pjn('checkpoints', self.tbx_wrtr_dir)):
                os.mkdir(pjn('checkpoints', self.tbx_wrtr_dir))

            fpath=pjn('checkpoints', self.tbx_wrtr_dir, nm)

            d = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(d, fpath)

def main(args):
    # for deterministic training, enable all below options.
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = False
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    wandb.init(project="DL20")
    orig_cwd = os.getcwd()
    
    # bring effi model from this : https://github.com/lukemelas/EfficientNet-PyTorch
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()

    additional_cfg = {'device': None}
    additional_cfg['device'] = torch.device('cuda')

    trainable_params = [
        {'params': list(filter(lambda p:p.requires_grad, model.parameters())), 'lr':args.lr},
    ]

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9)

    scaler = torch.cuda.amp.GradScaler() 

    if args.pretrained_ckpt:
        print(f"  Using pretrained model only and its checkpoint "
              f"'{args.pretrained_ckpt}'")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
        print("load optimizer's params")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        optimizer.load_state_dict(loaded_struct['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    now = datetime.datetime.now()
    ti = str(now.strftime('%Y-%m-%d-%H-%M-%S'))
    additional_cfg['tbx_wrtr_dir'] = os.getcwd() + "/checkpoints/" + str(ti)

    wandb.run.name = str(ti)
    label_loader, unlabel_loader, val_loader = init(
        args.batch_size_train, args.batch_size_val, args.ratio, args
    )


    trainer = TrainManager(
        model,
        optimizer,
        args,
        additional_cfg,
        label_loader=label_loader,
        unlabel_loader=unlabel_loader,
        val_loader=val_loader,
        scaler=scaler,
        num_classes=20
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')

    parser.add_argument('--batch-size-train', type=int, default=16,    
                        help='Batch size for train data (default: 16)')
    parser.add_argument('--batch-size-val', type=int, default=4,
                        help='Batch size for val data (default: 4)')
    parser.add_argument('--ratio', type=float, default=0.02,
                        help="label:unlabel ratio(0.5, 0.125, 0.05, 0.02)")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="prediction const loss coefficient")
    parser.add_argument('--beta', type=float, default=0.2,
                        help="gradcam const loss coefficient")

    parser.add_argument('--save-ckpt', type=int, default=5,
                        help='number of epoch save current weight? (default: 5)')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='start epoch (default: 0)')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='end epoch (default: 30)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--lr-anneal-rate', type=float, default=0.995,
                        help='Annealing rate (default: 0.95)')
    parser.add_argument('--upscale-factor', type=int, default=8,
                        help='Upscale factor for bilinear upsampling')
    

    args = parser.parse_args()
    main(args)