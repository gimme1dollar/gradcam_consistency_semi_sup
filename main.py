from efficientnet_pytorch.model import EfficientNet
import torch
import torch.multiprocessing
from torchvision import transforms
from dataset.dataloader import LoadDataset, LoadSemiDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torchvision import datasets, transforms

from tqdm import tqdm
from os.path import join as pjn
import os.path, os, datetime, time, random
import wandb, argparse
from utils.losses import *
from utils.visualize import *
import math
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as VF
from typing import Callable
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

cifar10_mean = [0.4913, 0.4821, 0.4465]
cifar10_std = [0.2470, 0.2434, 0.2615]

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = None
        self.activations = None
        self.reshape_transform = reshape_transform
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
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=True,
                 reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)
        self.upscale_layer = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        
    def forward(self, input_img):
        return self.model(input_img)
    
    def buffer_clear(self):
        self.activations_and_grads.buffer_clear()

    def __call__(self, input_tensor, expand):
        # input_tensor : b x c x h x w
        self.buffer_clear()
        self.model.eval()
        
        cam_stack=[]    
        for batch_idx in tqdm(range(input_tensor.shape[0]), desc='cam_calc', leave=False): # batch 개
            self.model.zero_grad()
            output = self.activations_and_grads(input_tensor[batch_idx].unsqueeze(0)) # 1 x c x h x w

            y_c = output[batch_idx, torch.argmax(output)] #arg_max # h x w ; GAP over channel
            y_c.backward(retain_graph=True) 
            activations = self.activations_and_grads.activations
            grads = self.activations_and_grads.gradients
            self.buffer_clear()
            weights = torch.mean(grads, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations, dim=0)

            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=expand.size()[2:], mode='bilinear', align_corners=True)

            min_v = torch.min(cam)
            range_v = torch.max(cam) - min_v
            if range_v > 0:
                cam = (cam - min_v) / range_v
            else:
                cam = torch.zeros(cam.size())
            cam_stack.append(cam.cuda())
            
            self.activations_and_grads.clear_list()
            del y_c, activations, grads, weights, cam, output
            torch.cuda.empty_cache()
        concated_cam = torch.cat(cam_stack, dim=0).squeeze() # b x 5 x h x w
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


def init(val_batch, val_split, args):

    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug

    transform_cifar10 = transforms.Compose([transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    
    label_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=val_batch, sampler=label_sampler)
    unlabel_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=val_batch, sampler=unlabel_sampler) 
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=val_batch, shuffle=True)
    
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
        self.upsampler = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        for idx, module in enumerate(self.model.modules()):
            #if idx > 200:
            #    print(idx, module)
            if idx == 479:
                target_layer = module
        self.get_cam = GradCAM(model=self.model, target_layer=target_layer, use_cuda=False)

        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.resize_512_transform = transforms.Compose([
            transforms.Resize((512, 512))
        ])
        self.resize_256_transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])
        self.resize_64_transform = transforms.Compose([
            transforms.Resize((64, 64))
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

    def color_augmentation(self, i, img):
        color_transform = transforms.Compose([
            transforms.ColorJitter(i, i, i, i)
        ])

        return color_transform(img)

    def get_crop_params(self, img):
        w_, h_ = img.size(2), img.size(3)
        xl = random.randint(0, w_ / 8)
        xr = 0
        while (((xr - xl) < (w_ * 7 / 8)) and (xr <= xl)):
            xr = random.randint(xl, w_)

        yl = random.randint(0, h_ / 8)
        yr = 0
        while (((yr - yl) < (h_ * 7 / 8)) and (yr <= yl)):
            yr = random.randint(yl, h_)

        return xl, yl, xr, yr

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

    def train(self):
        transform_unlabel = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()
        ])

        start = time.time()
        epoch = 0
        iter_per_epoch = len(self.label_loader)
        print("  iteration per epoch(considered batch size): ", iter_per_epoch)
        print("  label iter : ", len(self.label_loader))
        print("  unlabel iter : ", len(self.unlabel_loader))
        print("  val iter : ", len(self.val_loader))
        print("  image iter : ", len(self.unlabel_loader))
        print("  Progress bar for training epochs:")
        end_epoch = self.args.start_epoch + self.args.num_epochs

        unlabel_dataloader = iter(cycle(self.unlabel_loader))
        alpha = 0.965
        p_cutoff = 0.80
        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):

            if epoch % 1 == 0:
                top1_acc, top3_acc, top5_acc = self.validate(self.model, self.add_cfg['device'])
                wandb.log({"validation/top1_acc" : top1_acc, "validation/top3_acc" : top3_acc, "validation/top5_acc" : top5_acc})
                top1_acc_stu = top1_acc

                #top1_acc, top3_acc, top5_acc = self.validate(self.teacher ,self.add_cfg['device'])
                #wandb.log({"validation/teacher_top1_acc" : top1_acc, "validation/teacher_top3_acc" : top3_acc, "validation/teacher_top5_acc" : top5_acc})
                #top1_acc_t = top1_acc

            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            for t_idx, (image, target) in tqdm(enumerate(self.label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                image = self.upsampler(image)
                image = image.to(self.add_cfg['device']) # DL20
                target = target.to(self.add_cfg['device'])

                self.optimizer.zero_grad()
                losses_list = []
                with torch.cuda.amp.autocast():
                    ### sup loss ###
                    outputs = self.model(image)
                    celoss = CEloss(outputs, target)
                    losses_list.append(celoss)   
                wandb.log({"training/celoss" : celoss})

                image_ul = next(unlabel_dataloader)
                images = []
                for i in image_ul[0]:
                    #i = transform_unlabel(i)
                    images.append(i)
                image_ul = torch.stack(images)
                image_ul = image_ul.to(self.add_cfg['device']) # DL20
                image_ul = self.upsampler(image_ul)
                
                ## Augmentation
                wk_image = self.color_augmentation(0.1, image_ul)
                wk_image = wk_image.cuda()
                
                wk_label = self.model(image_ul)
                wk_prob = torch.softmax(wk_label, dim=-1)
                max_probs, max_idx = torch.max(wk_prob, dim=-1)
                mask_p = max_probs.ge(p_cutoff).float()
                mask_p = mask_p.cpu().detach().numpy()

                i, j, h, w = self.get_crop_params(image_ul)
                st_image = VF.crop(image_ul, i, j, h, w)
                st_image = self.color_augmentation(0.5, st_image)
                st_image = self.resize_256_transform(st_image)
                st_image = st_image.cuda()
                
                if t_idx % 50 == 0:
                    visualize_rescale_image(cifar10_mean, cifar10_std, image_ul, "imagenet_org/imagenet")
                    visualize_rescale_image(cifar10_mean, cifar10_std, wk_image, "imagenet_wk/imagenet")
                    visualize_rescale_image(cifar10_mean, cifar10_std, st_image, "imagenet_st/imagenet")

                ## Getting cam
                wk_cam = []
                for img in wk_image:
                    img = img.unsqueeze(0)
                    wk_cam_ = self.get_cam(img, img)
                    wk_cam.append(wk_cam_)
                wk_cam = torch.stack(wk_cam)
                if t_idx % 50 == 0:
                    visualize_cam(wk_image, wk_cam, cifar10_mean, cifar10_std, "wk_cam/imagenet")   
         
                st_cam = []
                for img in st_image:
                    img = img.unsqueeze(0)
                    #img = upscale_layer(img)
                    st_cam_ = self.get_cam(img, img)
                    st_cam.append(st_cam_)
                st_cam = torch.stack(st_cam)
                if t_idx % 50 == 0:       
                    visualize_cam(st_image, st_cam, cifar10_mean, cifar10_std, "st_cam/imagenet") 

                gt_cam = VF.crop(wk_cam, i, j, h, w)
                gt_cam = self.resize_256_transform(gt_cam)
                if t_idx % 50 == 0:       
                    visualize_cam(st_image, gt_cam, cifar10_mean, cifar10_std, "gt_cam/imagenet")    
                del wk_cam
                
                ## Get gradcam_consistency_loss
                mask = [ torch.ones_like(gt_cam[0]) if int(mask_p[i]) else torch.zeros_like(gt_cam[0]) for i in range(gt_cam.size(0))]
                mask = torch.stack(mask)
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    mseloss = MSEloss(st_cam * mask, gt_cam * mask)
                    if math.isnan(mseloss) is False:
                        losses_list.append(mseloss)
                    wandb.log({"training/mseloss" : mseloss})
                

                ## Train model
                self.model.train()
                t_loss = total_loss(losses_list)
                wandb.log({"training/tloss" : t_loss})

                t_loss = total_loss(losses_list)
                self.scaler.scale(t_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.adjust_learning_rate(epoch)
            self.save_ckpt(epoch)
            
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
        args.batch_size_val, args.ratio, args
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
    parser.add_argument('--batch-size-val', type=int, default=16,
                        help='Batch size for val data (default: 4)')
    parser.add_argument('--ratio', type=float, default=0.02,
                        help="label:unlabel ratio(0.5, 0.125, 0.05, 0.02)")

    parser.add_argument('--save-ckpt', type=int, default=5,
                        help='number of epoch save current weight? (default: 5)')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='start epoch (default: 0)')
    parser.add_argument('--num-epochs', type=int, default=1200,
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