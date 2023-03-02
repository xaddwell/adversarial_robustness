import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random
from torchvision.datasets import MNIST,CIFAR10,CIFAR100,SVHN,ImageFolder
from default_config import get_default_cfg
from utils.transform import *

cfg = get_default_cfg()

torch.manual_seed(cfg.random_seed)


def get_loader(datasets_name, stage, batch_size, num_workers, transform = to_tensor):

    if datasets_name == 'ImageNet':
        dataset = ImageFolder(root=os.path.join(cfg.datasets_dir,'ImageNet'),transform=transform)
        dataset_size  = len(dataset)
        train_size = int(dataset_size*(1 - cfg.test_split))
        test_size = dataset_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        if stage == 'test':

            return DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

        else:
            val_size = int(train_size * cfg.val_split)
            train_size = train_size - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            return DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers),\
                   DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)


    elif datasets_name in ['CIFAR10','CIFAR100']:

        if stage == 'test':

            if datasets_name == "CIFAR10":
                dataset = CIFAR10(root=os.path.join(cfg.datasets_dir,"CIFAR10"),train = True,transform = transform)
            elif datasets_name == "CIFAR100":
                dataset = CIFAR100(root=os.path.join(cfg.datasets_dir,"CIFAR100"),train = True,transform = transform)

            test_dataset, atk_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.95), len(dataset) - int(len(dataset)*0.95)])

            return DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers),\
                   DataLoader(atk_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)

        else:

            if datasets_name == "CIFAR10":
                dataset = CIFAR10(root=os.path.join(cfg.datasets_dir,"CIFAR10"),train = True,transform = transform)
            elif datasets_name == "CIFAR100":
                dataset = CIFAR100(root=os.path.join(cfg.datasets_dir,"CIFAR100"),train = True,transform = transform)

            dataset_size = len(dataset)
            train_size = int(dataset_size * (1 - cfg.val_split))
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            return DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers), \
                   DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    else:
        raise ValueError('datasets {} not exist'.format(datasets_name))


class ImageNet_datasets(Dataset):
    def __init__(self, filename, shuffle=False):
        print("=====>>>load ImageNet_datasets from {}".format(filename))
        ori_dir = filename + '/ori/'
        class_list=os.listdir(ori_dir)
        self.class_file=[]
        for name in class_list:
            temp_file = ori_dir +name
            label = name.split('.')[0].split('_')[-1]
            advs_dir = filename + '/advs/' + name
            self.class_file.append((temp_file,advs_dir,label))
        if shuffle:
            random.shuffle(self.class_file)

    def __getitem__(self, idx):
        ori,advs,label =self.class_file[idx]
        ori = to_tensor(Image.open(ori))
        advs = to_tensor(Image.open(advs))
        label = torch.tensor(int(label))
        return ori,advs,label

    def __len__(self):
        return len(self.class_file)


class compare_imageNet_datasets(Dataset):
    def __init__(self, filename, shuffle=True,transform=None):
        print("=====>>>load compare_imageNet_datasets from {}".format(filename))
        ori_dir = filename + '/ori/'
        class_list=os.listdir(ori_dir)
        self.transform = transform
        self.class_file=[]
        for name in class_list:
            temp_file = ori_dir +name
            label = name.split('.')[0].split('_')[-1]
            advs_dir = filename + '/advs/' + name
            self.class_file.append((temp_file,advs_dir,label))
        if shuffle:
            random.shuffle(self.class_file)

    def __getitem__(self, idx):
        ori,advs,label =self.class_file[idx]
        if self.transform == None:
            ori = to_tensor(Image.open(ori))
            advs = to_tensor(Image.open(advs))
        else:
            ori = self.transform(Image.open(ori))
            advs = self.transform(Image.open(advs))
        label = torch.tensor(int(label))
        return ori,advs,label

    def __len__(self):
        return len(self.class_file)

class imageNet_datasets(Dataset):
    def __init__(self, filename,transform,shuffle=False):
        class_list=os.listdir(filename)
        self.class_file=[]
        for name in class_list:
            temp_file=os.path.join(filename,name)
            for item in os.listdir(temp_file):
                if not item.endswith('json'):
                    img_dir=os.path.join(temp_file,item)
                    self.class_file.append((img_dir,name))
        if shuffle:
            random.shuffle(self.class_file)
        self.transform = transform

    def __getitem__(self, idx):
        img,label=self.class_file[idx]
        img = self.transform(Image.open(img))
        label = torch.tensor(int(label))
        return img,label

    def __len__(self):
        return len(self.class_file)

