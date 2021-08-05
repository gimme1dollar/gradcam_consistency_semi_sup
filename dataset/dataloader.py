import os
import torch
import glob
import numpy as np
from os.path import join as pjn

from PIL import Image
from typing import Union

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

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
    
class DL20_dataset(Dataset):
    def __init__(self, data_path, transform, mode='valid'):
        super(DL20_dataset, self).__init__()
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

class DL20_semi_dataset(Dataset):
    def __init__(self, data_path, transform, mode='label', ratio=0.05):
        super(DL20_semi_dataset, self).__init__()
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
            return img, label
        else: 
            raise NotImplementedError()


def init(exp_data, train_batch, val_batch, test_batch, args):

    if exp_data == 'dl20':
        data_path = pjn(os.getcwd(), "dataset", "DL20")

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

        label_dataset = DL20_semi_dataset(data_path = data_path, transform=transform_train , mode='label', ratio=args.ratio)
        unlabel_dataset = DL20_semi_dataset(data_path = data_path, transform=transform_val , mode='unlabel', ratio=args.ratio)
        val_dataset = DL20_dataset(data_path = data_path, transform=transform_val , mode='valid')
        test_dataset = DL20_dataset(data_path = data_path, transform=transform_test , mode='test')

        label_loader = torch.utils.data.DataLoader(
                dataset=label_dataset, batch_size=train_batch,
                num_workers=4, shuffle=True, pin_memory=True
        )

        unlabel_loader = torch.utils.data.DataLoader(
                dataset=unlabel_dataset, batch_size=train_batch,
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

    if exp_data == 'cifar10':
        transform_cifar10 = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=True, download=True, transform=transform_cifar10)
        test_dataset = datasets.CIFAR10(root='./dataset/Cifar10', train=False, download=True, transform=transform_cifar10)

        split_ratio = args.ratio
        shuffle_dataset = True
        random_seed= 123

        dataset_size = len(train_dataset)
        trainset_size = int( dataset_size * 4 / 5)

        indices = list(range(dataset_size))
        split = int(np.floor(split_ratio * trainset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        label_indices, unlabel_indices, valid_indices = \
                    indices[:split], indices[split:trainset_size], indices[trainset_size:]

        label_sampler = SubsetRandomSampler(label_indices)
        unlabel_sampler = SubsetRandomSampler(unlabel_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        label_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch, sampler=label_sampler)
        unlabel_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch, sampler=unlabel_sampler) 
        val_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=val_batch, sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=True)

    return label_loader, unlabel_loader, val_loader, test_loader