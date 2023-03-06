# partial code is refered from https://stackoverflow.com/questions/51511074/how-to-create-sub-network-reference-in-pytorch3.


from __future__ import print_function, division
import os
from .network import vgg16_ori
import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import matplotlib
import copy

matplotlib.use('Agg')
from models.eca_module import eca_layer

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def _vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def _vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg16_ori(is_freeze=True):
    # Load the pretrained model from pytorch
    model = _vgg16(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model


def vgg16_bn_ori(is_freeze=True):
    # Load the pretrained model from pytorch
    model = _vgg16_bn(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model


class VGG_Freeze_conv_FC2_attention(nn.Module):
    """
    VGG-16 based network. The weights except for the last layer is frozen.
    In addition, the output label is replaced as single node output.
    The single output predicts the valence of the natural input.

    """

    def __init__(self):
        super(VGG_Freeze_conv_FC2_attention, self).__init__()
        """
        Args:
        """
        vgg = vgg16_ori()

        for param in vgg.features.parameters():
            param.requires_grad = False

        for param in vgg.classifier.parameters():
            param.requires_grad = True

        # Newly created modules have require_grad=True by default

        self.features = vgg.features
        self.ECA_Module = eca_layer()
        self.classifier = vgg.classifier[:-1]

        self.fully_connected_layers = nn.Sequential(nn.Linear(4096, 1024),
                                                    nn.ReLU(True),
                                                    nn.Dropout(),
                                                    nn.Linear(1024, 1024),
                                                    nn.ReLU(True),
                                                    nn.Dropout(),
                                                    nn.Linear(1024, 1),
                                                    nn.Sigmoid())

    def forward(self, x):
        input_image = x
        y_features = self.features(input_image)
        y_attention = self.ECA_Module(y_features)
        y_attention = torch.flatten(y_attention, start_dim=1)
        y_classifier = self.classifier(y_attention)
        y = self.fully_connected_layers(y_classifier)
        return y


class VGG_Freeze_conv1_FC2_attention(nn.Module):
    """
        VGG-16 based network. The weights except for the last layer is frozen.
        In addition, the output label is replaced as single node output.
        The single output predicts the valence of the natural input.

        Thus, this model only has conv1 part of the VGG as a visual cortex.
        """

    def __init__(self, path, model):
        super(VGG_Freeze_conv1_FC2_attention, self).__init__()
        """
        Args:
        """
        old_model = torch.load(os.path.join(path, model), map_location='cuda:0')
        base_model = old_model['model']
        base_model.load_state_dict(old_model['state_dict'])

        for param in base_model.features.parameters():
            param.requires_grad = False

        for param in base_model.classifier.parameters():
            param.requires_grad = True

        # Newly created modules have require_grad=True by default
        self.base_model = base_model
        self.features = base_model.features[0:9]
        self.ECA_Module = eca_layer()
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8, padding=0, dilation=1, ceil_mode=False)
        self.classifier = base_model.classifier[:-1]

        #self.classifier[0] = nn.Linear(7058, 4096)
        self.fully_connected_layers = nn.Sequential(nn.Linear(4096, 1024),
                                                    nn.ReLU(True),
                                                    nn.Dropout(),
                                                    nn.Linear(1024, 1024),
                                                    nn.ReLU(True),
                                                    nn.Dropout(),
                                                    nn.Linear(1024, 1),
                                                    nn.Sigmoid())

    def forward(self, x):
        input_image = x
        y_features = self.features(input_image)
        y_attention = self.ECA_Module(y_features)
        y_pool = self.pool(y_attention)
        y_attention = torch.flatten(y_pool, start_dim=1)
        y_classifier = self.classifier(y_attention)
        y = self.fully_connected_layers(y_classifier)
        return y