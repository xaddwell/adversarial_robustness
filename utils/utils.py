import torch
from models import ResNet18_30,Mobilenet_v2_30
from models import ShuffleNet_v2_30,Densenet121_30
from models.Unet import ResUnet,ResUnetPlusPlus,ResUnet01
from timm import create_model
from default_config import get_default_cfg

cfg = get_default_cfg()


__all__ = ['get_classifier','get_generator']


def get_classifier(args,pretrained = True, feature_map = False):

    model_name = args.model_name
    dataset_name = args.datasets

    if dataset_name == "ImageNet":

        if model_name == 'ResNet18':
            model = ResNet18_30(pretrained=pretrained,feature_map=feature_map)
        elif model_name == 'ShuffleNetv2':
            model = ShuffleNet_v2_30(pretrained=pretrained,feature_map=feature_map)
        elif model_name == 'MobileNetv2':
            model = Mobilenet_v2_30(pretrained=pretrained,feature_map=feature_map)
        elif model_name == 'DenseNet121':
            model = Densenet121_30(pretrained=pretrained,feature_map=feature_map)
        elif model_name == 'ViT-patch16':
            model = create_model(model_name = 'vit_base_patch16_224')
        elif model_name == 'Inception-ResNet-v2':
            model = create_model(model_name = 'inception_resnet_v2')

    elif dataset_name == "CIFAR10":

        if model_name == 'MobileNetv2':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_4", pretrained=True)
        elif model_name == 'ShuffleNetv2':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x1_5", pretrained=True)
        elif model_name == 'ResNet':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    elif dataset_name == "CIFAR100":

        if model_name == 'MobileNetv2':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
        elif model_name == 'ShuffleNetv2':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_shufflenetv2_x1_5", pretrained=True)
        elif model_name == 'ResNet':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)

    return model

def get_generator(victim_model,source_attack_method,
                  generator_name,use_cuda=True):

    if generator_name == 'ResUNet':
        model = ResUnet(channel=3)
    elif generator_name == 'ResUNetPlusPlus':
        model = ResUnetPlusPlus(channel=3)
    elif generator_name == 'ResUNet01':
        model = ResUnet01(channel=3)

    generator_dir = cfg.ckpt_dir + '/{}/{}/{}.pt'. \
        format(victim_model, source_attack_method, generator_name)

    if generator_name:
        print("=====>>>load pretrained model {} from {}".
              format(generator_name, generator_dir))
        model.load_state_dict(torch.load(
            generator_dir,map_location='cuda' if use_cuda else 'cpu'))
        return model
    else:
        return None

def model_load_ckpt_eval(args,model):
    ckpt_path = "{}/{}_{}_ckpt_best".format(args.ckpt_dir,args.datasets,args.model_name)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['net'])
    model.eval()

def get_logger(filename):

    return Logger(log_path=filename)


class Logger(object):
    def __init__(self, log_path="default.log"):
        import sys
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1, encoding="utf-8")

    def print(self, *message):
        message = ",".join([str(it) for it in message])
        self.terminal.write(str(message) + "\n")
        self.log.write(str(message) + "\n")

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()