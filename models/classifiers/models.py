import torchvision
from torch.nn import Linear
import torch
from torch import nn
from .resnet import resnet18
from .mobilenet import mobilenet_v2
from .shufflenet import shufflenet_v2_x1_0

__all__ = ['ResNet18_30','Mobilenet_v2_30','Densenet121_30','ShuffleNet_v2_30']

class ResNet18_30(nn.Module):
    def __init__(self,pretrained=True,feature_map = False):
        super(ResNet18_30, self).__init__()
        self.model = resnet18(pretrained=pretrained,feature_map = feature_map)
        self.model.fc.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map

        x = self.model(x)
        return x


class Mobilenet_v2_30(nn.Module):
    def __init__(self,pretrained=True,feature_map = False):
        super(Mobilenet_v2_30, self).__init__()
        self.model= mobilenet_v2(pretrained=pretrained,feature_map = feature_map)
        self.model.classifier.add_module(name='2',module=nn.Dropout(p=0.2,inplace=False))
        self.model.classifier.add_module(name='3', module=nn.Linear(in_features=1000,out_features=30))
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map
        x=self.model(x)
        return x


class ShuffleNet_v2_30(nn.Module):
    def __init__(self,pretrained=True,feature_map = False):
        super(ShuffleNet_v2_30, self).__init__()
        self.model= shufflenet_v2_x1_0(pretrained = pretrained,feature_map = feature_map)
        self.model.fc.out_features = 30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map

        x = self.model(x)
        return x


class Densenet121_30(nn.Module):
    def __init__(self,pretrained=True,feature_map = False):
        super(Densenet121_30, self).__init__()
        self.model=torchvision.models.densenet121(pretrained = pretrained,feature_map = feature_map)
        self.model.classifier.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map
        x = self.model(x)
        return x



