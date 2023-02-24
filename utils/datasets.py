import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random
from torchvision.datasets import MNIST,CIFAR10,CIFAR100,SVHN,ImageFolder
from default_config import get_default_cfg

cfg = get_default_cfg()

torch.manual_seed(cfg.random_seed)

to_tensor = transforms.Compose([transforms.ToTensor()])

svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

imagenet_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
])

cifar10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

cifar10_transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4940, 0.4850, 0.4504), (0.2467, 0.2429, 0.2616))
])

cifar100_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
])

cifar100_transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
])

mnist_transform_train = transforms.Compose([
    # 数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor,图像范围[0, 255] -> [0.0,1.0]
    transforms.ToTensor(),
    # 使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_transform_test = transforms.Compose([
    # 数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor,图像范围[0, 255] -> [0.0,1.0]
    transforms.ToTensor(),
    # 使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]
    transforms.Normalize((0.1326,), (0.3106,))
])


def get_loader(datasets_name, stage, batch_size, num_workers):

    if datasets_name == 'ImageNet':
        dataset = ImageFolder(root=os.path.join(cfg.datasets_dir,'ImageNet'),transform=imagenet_transform)
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


    elif datasets_name in ['MNIST','SVHN','CIFAR10','CIFAR100']:

        if stage == 'test':

            if datasets_name == "MNIST":
                dataset = MNIST(root=os.path.join(cfg.datasets_dir),train = True,transform = mnist_transform_test)
            elif datasets_name == "SVHN":
                dataset = SVHN(root=os.path.join(cfg.datasets_dir),split = "train",transform = svhn_transform)
            elif datasets_name == "CIFAR10":
                dataset = CIFAR10(root=os.path.join(cfg.datasets_dir),train = True,transform = cifar10_transform_test)
            elif datasets_name == "CIFAR100":
                dataset = CIFAR100(root=os.path.join(cfg.datasets_dir),train = True,transform = cifar100_transform_test)

            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

        else:

            if datasets_name == "MNIST":
                dataset = MNIST(root=os.path.join(cfg.datasets_dir),train = True,transform = mnist_transform_train)
            elif datasets_name == "SVHN":
                dataset = SVHN(root=os.path.join(cfg.datasets_dir),split = "train",transform = svhn_transform)
            elif datasets_name == "CIFAR10":
                dataset = CIFAR10(root=os.path.join(cfg.datasets_dir,"CIFAR10"),train = True,transform = cifar10_transform_train,download=True)
            elif datasets_name == "CIFAR100":
                dataset = CIFAR100(root=os.path.join(cfg.datasets_dir),train = True,transform = cifar100_transform_train)

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

