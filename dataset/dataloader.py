import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob
from PIL import Image
from typing import Union
import numpy as np

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
            return img, label
        else: 
            raise NotImplementedError()
