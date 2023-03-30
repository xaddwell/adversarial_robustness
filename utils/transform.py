import torch
from torchvision import transforms
from default_config import get_default_cfg

cfg = get_default_cfg()

torch.manual_seed(cfg.random_seed)

to_tensor = transforms.Compose([transforms.ToTensor()])

diff2raw = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
raw2diff = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
cifar10_raw2clf = transforms.Normalize((0.4940, 0.4850, 0.4504), (0.2467, 0.2429, 0.2616))
cifar100_raw2clf = transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
imagenet_raw2clf = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

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


def get_transform(model_name,datasets,stage):
    if model_name == "WideResNet28_10":
        transform = to_tensor
    elif stage == "train":
        if datasets == "CIFAR10":
            transform = cifar10_transform_train
        elif datasets == "CIFAR100":
            transform = cifar100_transform_train
        elif datasets == "ImageNet":
            transform = imagenet_transform
        else:
            raise ValueError("no match")
    elif stage == "test":
        if datasets == "CIFAR10":
            transform = cifar10_transform_test
        elif datasets == "CIFAR100":
            transform = cifar100_transform_test
        elif datasets == "ImageNet":
            transform = imagenet_transform
        else:
            raise ValueError("no match")
    else:
        raise ValueError("no match")

    return transform
