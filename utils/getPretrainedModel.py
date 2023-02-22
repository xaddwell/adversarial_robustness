import torch
from models import ResNet18_30,Mobilenet_v2_30,ResNet18_with_feature
from models import ShuffleNet_v2_30,Densenet121_30,ShuffleNet_with_feature
from models.core import ResUnet,ResUnetPlusPlus,ResUnet01
from models import Generator as cycleGAN_G
from config import *

def get_trained_classifier(model_name,use_cuda=True,feature_map = False):

    if model_name == 'ResNet18':
        model = ResNet18_30(feature_map=feature_map)
    elif model_name == 'ShuffleNetv2':
        model = ShuffleNet_v2_30(feature_map=feature_map)
    elif model_name == 'MobileNetv2':
        model = Mobilenet_v2_30(feature_map=feature_map)
    elif model_name == 'DenseNet121':
        model = Densenet121_30(feature_map=feature_map)
    elif model_name == 'ShuffleNetv2_with_allfea':
        model = ShuffleNet_with_feature(feature_map=feature_map)
        model_name = model_name.split("_")[0]
    elif model_name == 'ResNet18_with_allfea':
        model = ResNet18_with_feature(feature_map=feature_map)
        model_name = model_name.split("_")[0]


    model_dir = model_weight_dir + '/{}.pt'.format(model_name)

    if model_name:
        print("=====>>>load pretrained model {} from {}".
              format(model_name, model_dir))
        model.load_state_dict(torch.load(
            model_dir,map_location='cuda' if use_cuda else 'cpu'))
        return model
    else:
        return None

def get_trained_generator(victim_model,source_attack_method,
                  generator_name,use_cuda=True):

    if generator_name == 'ResUNet':
        model = ResUnet01(channel=3)
    elif generator_name == 'ResUNetPlusPlus':
        model = ResUnetPlusPlus(channel=3)
    elif generator_name == 'cycleGAN_G':
        model = cycleGAN_G(input_nc=3,output_nc=3)
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