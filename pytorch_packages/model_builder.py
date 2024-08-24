import torch
from torch import nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights


from pathlib import Path
import json

def load_model(model_name: str=None, class_names: None|list = None, pretrain_model_path: None|str= None, device: str= 'cpu'):
    info_data = None
    
    if pretrain_model_path is not None:
        pretrain_model_path = Path(pretrain_model_path)

        with open(pretrain_model_path / 'info.json', 'r') as f:
            info_data = json.load(f)

        class_names = info_data['class_names']
        model_name = info_data['model']
        
    if model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights= weights).to(device)
        
        features= model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features=features, out_features=len(class_names), bias=True)
        )
    elif model_name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights= weights).to(device)
        
        features= model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features=features, out_features=len(class_names), bias=True)
        )
    elif model_name == 'vgg19':
        weights = VGG19_Weights.DEFAULT
        model = vgg19(weights= weights).to(device)
        
        features= model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features=features, out_features=len(class_names), bias=True)
        )
    elif model_name == 'vgg16':
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights= weights).to(device)
        
        features= model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features=features, out_features=len(class_names), bias=True)
        )
    elif model_name == 'vgg11_bn':
        weights = VGG11_BN_Weights.DEFAULT
        model = vgg11_bn(weights= weights).to(device)
        
        features= model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features=features, out_features=len(class_names), bias=True)
        )
    elif model_name == 'effi_net_v2_s':
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights= weights).to(device)
        
        features= model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features=features, out_features=len(class_names), bias=True)
        )
        
    if pretrain_model_path is not None:
        model.load_state_dict(torch.load(f=pretrain_model_path / f'{model_name}.pth'))
            
    model = model.to(device)
            
    return model, info_data
