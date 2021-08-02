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
from dataset.dataloader import LoadDataset, LoadSemiDataset
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
from utils.losses import *
from utils.visualize import *
from PIL import Image
from typing import Union

import wandb, argparse
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_path = pjn(os.getcwd(), "dataset", "DL20")

def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]

def path_join(train_path, label, file_list):
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(train_path, label, f))
    
    return path_list

class LoadDataset(Dataset):
    def __init__(self, data_path, transform, mode='valid'):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        
        if mode == "test":
            self.test_load()
        else : 
            self.load_dataset()

    def test_load(self):
        root = os.path.join(self.data_path, self.mode)
        print("root : ", root)
        self.data = glob.glob(root+"/*.png")
        
    def load_dataset(self):
        train_path = os.path.join(self.data_path, self.mode)
        folder_list = os.listdir(train_path) # folder list [0,1,2,...,19]

        path_list = []
        for label_num in folder_list:
            file_path = os.path.join(train_path, label_num)     
            file_list = os.listdir(file_path)
            path_list += path_join(train_path, label_num, file_list)
        self.image_len = len(path_list)
        img_seq = [string_to_sequence(s) for s in path_list]
        self.image_v, self.image_o = pack_sequences(img_seq)

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        if self.mode == "test":
            img = Image.open(self.data[index]).convert('RGB')
            img = self.transform(img)
            return img
        else:
            path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
            label = int(path.split("/")[-2])
            img = Image.open(path).convert('RGB')
            img = self.transform(img)

            return img, label
            
class LoadSemiDataset(Dataset):
    def __init__(self, data_path, transform, mode='label', ratio=0.05):
        super(LoadSemiDataset, self).__init__()
        self.data_path = data_path
        self.list_name = str(ratio)+"_"+mode+"_path_list.txt"
        self.mode = mode
        self.transform = transform
        
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path, self.list_name)
        print(root)
        with open(os.path.join(self.data_path, self.list_name), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.image_len = len(file_names)
        img_seq = [string_to_sequence(s) for s in file_names]
        self.image_v, self.image_o = pack_sequences(img_seq)

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        path = sequence_to_string(unpack_sequence(self.image_v, self.image_o, index))
        label = int(path.split("/")[-2])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if self.mode == 'label':
            return img, label
        elif self.mode == 'unlabel':
            return img
        else: 
            raise NotImplementedError()

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def init(train_batch, val_batch, test_batch, args):

    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug

    transform_train = T.Compose([
        #T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    label_dataset = LoadSemiDataset(data_path = data_path, transform=transform_train , mode='label', ratio=args.ratio)
    unlabel_dataset = LoadSemiDataset(data_path = data_path, transform=transform_train , mode='unlabel', ratio=args.ratio)
    val_dataset = LoadDataset(data_path = data_path, transform=transform_val , mode='valid')
    test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')

    label_loader = torch.utils.data.DataLoader(
            dataset=label_dataset, batch_size=train_batch,
            num_workers=4, shuffle=True, pin_memory=True
    )

    unlabel_loader = torch.utils.data.DataLoader(
            dataset=unlabel_dataset, batch_size=4,
            num_workers=4, shuffle=True, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=val_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=test_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )

    return label_loader, unlabel_loader, val_loader, test_loader
    
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
        self.upsampler = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

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
        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):
            self.model.train()

            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            for t_idx, (image, target) in tqdm(enumerate(self.label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                image = self.upsampler(image)
                image = image.to(self.add_cfg['device']) # DL20
                target = target.to(self.add_cfg['device'])

                image_ul = next(unlabel_dataloader)
                image_ul = image_ul.to(self.add_cfg['device']) # DL20

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

            if epoch % 30 == 1:
                self.save_ckpt(epoch)

            if epoch % 5 == 1:
                top1_acc, top3_acc, top5_acc = self.validate(self.model, self.add_cfg['device'])
                wandb.log({"validation/top1_acc" : top1_acc, "validation/top3_acc" : top3_acc, "validation/top5_acc" : top5_acc})
                top1_acc_stu = top1_acc
            
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
    label_loader, unlabel_loader, val_loader, _ = init(
        args.batch_size_train, args.batch_size_val, args.batch_size_test, args
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
    parser.add_argument('--batch-size-test', type=int, default=1,
                        help='Batch size for test data (default: 128)')
    parser.add_argument('--ratio', type=float, default=0.125)

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
    

    
    args = parser.parse_args()
    main(args)