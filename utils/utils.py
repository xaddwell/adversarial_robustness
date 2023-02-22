import torch
from models import ResNet18_30,Mobilenet_v2_30
from models import ShuffleNet_v2_30,Densenet121_30
from models.Unet import ResUnet,ResUnetPlusPlus,ResUnet01
from timm import create_model
from config import *


__all__ = ['get_classifier','get_generator']


def get_classifier(model_name,pretrained = False, use_cuda=True,feature_map = False):

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

    model_dir = model_weight_dir + '/{}.pt'.format(model_name)

    if model_name:
        print("=====>>>load pretrained model {} from {}".
              format(model_name, model_dir))
        model.load_state_dict(torch.load(
            model_dir,map_location='cuda' if use_cuda else 'cpu'))
        return model
    else:
        return None

def get_generator(victim_model,source_attack_method,
                  generator_name,use_cuda=True):

    if generator_name == 'ResUNet':
        model = ResUnet(channel=3)
    elif generator_name == 'ResUNetPlusPlus':
        model = ResUnetPlusPlus(channel=3)
    elif generator_name == 'ResUNet01':
        model = ResUnet01(channel=3)

    generator_dir = save_generator_weight + '/{}/{}/{}.pt'. \
        format(victim_model, source_attack_method, generator_name)

    if generator_name:
        print("=====>>>load pretrained model {} from {}".
              format(generator_name, generator_dir))
        model.load_state_dict(torch.load(
            generator_dir,map_location='cuda' if use_cuda else 'cpu'))
        return model
    else:
        return None