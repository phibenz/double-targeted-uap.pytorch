from __future__ import division

import os, shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from networks.resnet import resnet20, resnet56
from networks.mobilenet_v2 import mobilenet_v2
from networks.vgg_cifar import VGG

def get_network(model_arch, input_size, num_classes=1000, finetune=False):
    #### CIFAR-10 & CIFAR-100 models ####
    if model_arch == "resnet20":
        net = resnet20(input_size=input_size, num_classes=num_classes)
    elif model_arch == "resnet56":
        net = resnet56(input_size=input_size, num_classes=num_classes)
    elif model_arch == "vgg16_cifar":
        net = VGG('VGG16', num_classes=num_classes)
    elif model_arch == "vgg19_cifar":
        net = VGG('VGG19', num_classes=num_classes)
    #### ImageNet models ####
    elif model_arch == "alexnet":
        net = models.alexnet(pretrained=True)
    elif model_arch == "resnet18":
        net = models.resnet18(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_arch == "resnet50":
        net = models.resnet50(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
    elif model_arch == "vgg16":
        net = models.vgg16(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_arch == "vgg19":
        net = models.vgg19(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_arch == "resnet152":
        net = models.resnet152(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
    elif model_arch == "mobilenet_v2":
        net = mobilenet_v2(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.classifier[1].in_features
            net.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.classifier[1].in_features
            net.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_arch == "inception_v3":
        net = models.inception_v3(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            # Handle the auxilary net
            num_ftrs = net.AuxLogits.fc.in_features
            net.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.AuxLogits.fc.in_features
            net.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Network {} not supported".format(model_arch))
    return net

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = False

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==True, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def get_num_non_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==False, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
