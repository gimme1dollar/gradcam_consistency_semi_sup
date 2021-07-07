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
from torch.nn import functional as F
from torchvision import datasets, transforms

from tqdm import tqdm
from os.path import join as pjn
import os.path, os, datetime, time, sys
import wandb, argparse
from utils.losses import *
import math
import warnings

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

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
        self.upsampler = torch.nn.Upsample(scale_factor=args.upscale_factor, mode='bilinear', align_corners=True)

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
                
            model_top1, model_top3, model_top5 = (correct_1 / total) * 100, (correct_3 / total) * 100, (correct_5 / total) * 100
        return (model_top1, model_top3, model_top5)


    def train(self):
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

        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):

            if epoch % 5 == 0:
                top1_acc, top3_acc, top5_acc = self.validate(self.model, self.add_cfg['device'])
                wandb.log({"validation/top1_acc" : top1_acc, "validation/top3_acc" : top3_acc, "validation/top5_acc" : top5_acc})

            self.model.train()
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
        model.load_state_dict(loaded_struct['model_state_dict'], strict=False)
        print("load optimizer's params")
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
