# partial code is refered from https://stackoverflow.com/questions/51511074/how-to-create-sub-network-reference-in-pytorch3.


from __future__ import print_function, division
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
    r"""VGG
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    This is base model. It receives the output from "make_layers" function for feature extractor.
    Args:
        features (nn.Module): This has to be the feature extractor module from the VGG.
        num_classes (int): Number of output variables.
        init_weights (bool): If True, it re-initialize the weights before training.
    """
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
    r"""make_layers
        This builds the feature extractor for the VGG models. Please refer to the configurations.
        Args:
            cfg (dict): Key with configuration code and Value with layer numbers.
            batch_norm (bool): If True, BatchNorm 2D is applied between Conv and Relu layer.
        """
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
    r"""VGG
        `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
        Function of building base model with VGG class
        Args:
            arch (str): String value which contains a key for the model url in Pytorch.
            cfg (dict): Key with configuration code and Value with layer numbers.
            batch_norm (bool): If True, BatchNorm 2D is applied between Conv and Relu layer.
            pretrained (bool): If True, imports the pretrained model from Pytorch based on the 'arch' key value
            progress (bool): If True, displays a progress bar of the download to stderr
        """
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
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def _vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg16_ori(is_freeze=True):
    r"""VGG 16-layer model (configuration "D") with layers frozen
        Args:
            is_freeze (bool): If True, the model parameters are all frozen.
        """
    model = _vgg16(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze
    return model


def vgg16_bn_ori(is_freeze=True):
    r"""VGG 16-layer model (configuration "D") with batch normalization and layer frozen
        Args:
            is_freeze (bool): If True, the model parameters are all frozen.
        """
    # Load the pretrained model from pytorch
    model = _vgg16_bn(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model


class VGG_Freeze_conv(nn.Module):
    r"""VGG 16-layer model (configuration "D") with only feature extractor frozen.
        Args:
            is_freeze (bool): If True, the model parameters in feature extractor modules are frozen.
        """

    def __init__(self):
        super(VGG_Freeze_conv, self).__init__()
        self.vgg = vgg16_ori()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Newly created modules have require_grad=True by default
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1),
                                               nn.Sigmoid())

    def forward(self, x):
        return self.vgg(x)


class VGG_BN_Freeze_conv(nn.Module):
    r"""VGG 16-layer model (configuration "D") with batch normalization & only feature extractor frozen.
        Args:
        """

    def __init__(self):
        super(VGG_BN_Freeze_conv, self).__init__()
        self.vgg = vgg16_bn_ori()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Newly created modules have require_grad=True by default
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1),
                                               nn.Sigmoid())

    def forward(self, x):
        return self.vgg(x)


class VGG_Freeze_conv_FC1(nn.Module):
    r"""VGG 16-layer model (configuration "D") with additional fully connected layer & only feature extractor frozen.
            Args:
            """

    def __init__(self):
        super(VGG_Freeze_conv_FC1, self).__init__()
        self.vgg = vgg16_ori()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Newly created modules have require_grad=True by default
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1024),
                                               nn.ReLU(True),
                                               nn.Dropout(),
                                               nn.Linear(1024, 1),
                                               nn.Sigmoid())

    def forward(self, x):
        return self.vgg(x)


class VGG_Freeze_conv_FC2(nn.Module):
    r"""VGG 16-layer model (configuration "D") with 2 additional fully connected layers & feature extractor frozen.
        Args:
        """

    def __init__(self):
        super(VGG_Freeze_conv_FC2, self).__init__()
        self.vgg = vgg16_ori()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Newly created modules have require_grad=True by default
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1024),
                                               nn.ReLU(True),
                                               nn.Dropout(),
                                               nn.Linear(1024, 1024),
                                               nn.ReLU(True),
                                               nn.Dropout(),
                                               nn.Linear(1024, 1),
                                               nn.Sigmoid())

    def forward(self, x):
        return self.vgg(x)


class Visual_Cortex_Amygdala(nn.Module):
    r"""Visual Cortex Amygdala model
        This Visual Cortex Amygdala model is the one type out of several variations, but final version.
        The model consists of 2 pathways, highroad (main road) and middleroad(shortcut pathway).

        High Road is the normal ventral stream pathway, which is the VGG-16 model without the last layer in this case.
        Middle Road is the shortcut pathway which models the affective system modulating the ventral visual pathway.

        The attention algorithm and shortcut pathway defined by the model was effective in reproducing behavioral
        results from the empirical experiments.

            Args:
                lowfea_VGGLayer (int): layer number index of where the shortcut pathway starts
                highfea_VGGLayer (int): depreciated
            """

    def __init__(self, lowfea_VGGlayer=10, highfea_VGGlayer=36):
        super(Visual_Cortex_Amygdala, self).__init__()

        # Importing the original vgg16 model, without batch normalization
        self.vgg = vgg16_ori()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # setting which layer in the model should be extracted for feature computation.
        self.lowfea_VGGlayer = lowfea_VGGlayer
        self.highfea_VGGlayer = highfea_VGGlayer

        # This section is defining the parts for the bottom-up attention module. (Middle Road)
        self.VGG_Middleroad = self.vgg.features[:self.lowfea_VGGlayer]

        # Max Pooling layers consisting the middle road.
        self.Middleroad_MaxPool = nn.MaxPool2d(kernel_size=29, stride=14, padding=0, dilation=1, ceil_mode=False)
        self.Global_MaxPool = nn.AdaptiveMaxPool2d(2)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=9, stride=5, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=13, stride=7, padding=0)

        # Adaptive Pooling layers consisting the middle road.
        self.Global_AdaptivePool1 = nn.AdaptiveAvgPool2d(1)
        self.Global_AdaptivePool2 = nn.AdaptiveAvgPool2d(2)

        # Efficient Channel Attention (ECA) Module.
        self.ECA_Module = eca_layer()
        Middleroad_FC_input_size = 1024

        # This section is the Fully Connected Layers which computes the high dimensional features computed based on the
        # bottom-up attention module.
        self.VCA_Middleroad = self.Middleroad_FC(Middleroad_FC_input_size)

        # This section is defining the parts for the high dimensional areas and modifying the last layer (High Road)
        self.VCA_Highroad = self.vgg
        self.VCA_Highroad.classifier = nn.Sequential(
            *list(self.vgg.children())[2][:-1])  # Filter all the layers except the last layer of VGG

        # High road output size is 4096, and Middle road output size is 512
        Highroad_Middleroad_FC_input_size = 4096 + 512

        # Last fully connected layers which computes the features from high road and low road
        self.VCA_FC = self.Highroad_Middleroad_FC(Highroad_Middleroad_FC_input_size)

    def Middleroad_FC(self, input_size):
        """
        :param input_size: in this model, it is pre-determined as 1024, because the concatenated output size of the
        bottom-up attention module is 1024.
        :return: the Sequential Layers for Middleroad fully connected layers.
        """
        Middleroad_FC = nn.Sequential(
            nn.Linear(input_size, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
        )
        return Middleroad_FC

    def Highroad_Middleroad_FC(self, input_size):
        """
        :param input_size: The output of the highroad is 4096 and middleroad is 512. Therefore, the input size is
        determined as 4608)
        :return: the Sequential Layers for the last fully connected network of the model.
        """
        Highroad_Middleroad_FC = nn.Sequential(
            nn.Linear(input_size, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        return Highroad_Middleroad_FC

    def forward(self, x):
        """
        :param x: receives the mini batch input with dimension of (N, C, H, W)
        :return: returns the raw output of VCA model. the intermediate output of each layer has the label of "y_" in
        front of layer name.
        """

        # saving the input prior to the computation
        input_image = x

        # computed output of the highroad
        y_Highroad_output = self.VCA_Highroad(input_image)

        # computed output of early layers in VGG-16 model to create input for bottom-up attention module.
        y_Middleroad = self.VGG_Middleroad(input_image)

        # compute the each part of bottom-up attention module
        y_Middleroad_MaxPool = self.Middleroad_MaxPool(y_Middleroad)
        y_Global_MaxPool = self.Global_MaxPool(y_Middleroad)
        y_MaxPool1 = self.MaxPool1(y_Middleroad)
        y_MaxPool2 = self.MaxPool2(y_Middleroad)
        y_MaxPool3 = self.MaxPool3(y_Middleroad)

        y_MaxPool1_Global_Adaptive_Pool2 = self.Global_AdaptivePool2(y_MaxPool1)
        y_MaxPool2_Global_Adaptive_Pool2 = self.Global_AdaptivePool2(y_MaxPool2)
        y_MaxPool3_Global_Adaptive_Pool2 = self.Global_AdaptivePool2(y_MaxPool3)

        eca_module_input = torch.cat([y_Global_MaxPool,
                                      y_MaxPool1_Global_Adaptive_Pool2,
                                      y_MaxPool2_Global_Adaptive_Pool2,
                                      y_MaxPool3_Global_Adaptive_Pool2], dim=1)

        y_eca_module = self.ECA_Module(eca_module_input)

        # flatten the tensors to concatenate them
        y_eca_module_output_Global_Adaptive_Pool1 = self.Global_AdaptivePool1(y_eca_module)

        # change the dimensionality of the 2 tensors before concatenation
        y_eca_module_output_Global_Adaptive_Pool1 = y_eca_module_output_Global_Adaptive_Pool1.view(
            y_eca_module_output_Global_Adaptive_Pool1.size(0), -1)
        y_Middleroad_MaxPool = y_Middleroad_MaxPool.view(y_Middleroad_MaxPool.size(0), -1)

        # concatenate the computed features from the attention module to feed the fully connected network in lowroad.
        y_Bottom_Up_Attention_Feature = torch.cat([y_Middleroad_MaxPool, y_eca_module_output_Global_Adaptive_Pool1], dim=1)

        # fully connected network of the lowroad.
        y_Middleroad_output = self.VCA_Middleroad(y_Bottom_Up_Attention_Feature)

        # concatenate the 1 dimensional feature tensor from high road and low road.
        Highroad_Middleroad_input = torch.cat([y_Highroad_output, y_Middleroad_output], dim=1)

        # last fully connected network to compute the high dimensional features based on the low road and high road
        # features.
        y_Highroad_Middleroad_FC = self.VCA_FC(Highroad_Middleroad_input)

        return y_Highroad_Middleroad_FC


class Visual_Cortex_Amygdala_wo_Attention(nn.Module):
    r"""Visual Cortex Amygdala model wo Attention
        This Visual Cortex Amygdala model is the one type out of several variations, but final version.
        The model consists of 2 pathways, highroad (main road) and middleroad(shortcut pathway).
        Only difference with the previous model is the existence of attention module.

        High Road is the normal ventral stream pathway, which is the VGG-16 model without the last layer in this case.
        Middle Road is the shortcut pathway which models the affective system modulating the ventral visual pathway.

        The attention algorithm and shortcut pathway defined by the model was effective in reproducing behavioral
        results from the empirical experiments.

            Args:
                lowfea_VGGLayer (int): layer number index of where the shortcut pathway starts
                highfea_VGGLayer (int): depreciated
            """

    def __init__(self, lowfea_VGGlayer=10, highfea_VGGlayer=36):
        super(Visual_Cortex_Amygdala_wo_Attention, self).__init__()

        # Importing the original vgg16 model, without batch normalization
        self.vgg = vgg16_ori()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # setting which layer in the model should be extracted for feature computation.
        self.lowfea_VGGlayer = lowfea_VGGlayer
        self.highfea_VGGlayer = highfea_VGGlayer

        # This section is defining the parts for the bottom-up attention module. (Middle Road)
        self.VGG_Middleroad = self.vgg.features[:self.lowfea_VGGlayer]

        # Max Pooling layers consisting the middle road.
        self.Middleroad_MaxPool = nn.MaxPool2d(kernel_size=29, stride=14, padding=0, dilation=1, ceil_mode=False)
        self.Global_MaxPool = nn.AdaptiveMaxPool2d(2)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=9, stride=5, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=13, stride=7, padding=0)

        # Adaptive Pooling layers consisting the middle road.
        self.Global_AdaptivePool1 = nn.AdaptiveAvgPool2d(1)
        self.Global_AdaptivePool2 = nn.AdaptiveAvgPool2d(2)

        # Efficient Channel Attention (ECA) Module.
        Middleroad_FC_input_size = 1024

        # This section is the Fully Connected Layers which computes the high dimensional features computed based on the
        # bottom-up attention module.
        self.VCA_Middleroad = self.Middleroad_FC(Middleroad_FC_input_size)

        # This section is defining the parts for the high dimensional areas and modifying the last layer (High Road)
        self.VCA_Highroad = self.vgg
        self.VCA_Highroad.classifier = nn.Sequential(
            *list(self.vgg.children())[2][:-1])  # Filter all the layers except the last layer of VGG

        # High road output size is 4096, and Middle road output size is 512
        Highroad_Middleroad_FC_input_size = 4096 + 512

        # Last fully connected layers which computes the features from high road and low road
        self.VCA_FC = self.Highroad_Middleroad_FC(Highroad_Middleroad_FC_input_size)

    def Middleroad_FC(self, input_size):
        """
        :param input_size: in this model, it is pre-determined as 1024, because the concatenated output size of the
        bottom-up attention module is 1024.
        :return: the Sequential Layers for middleroad fully connected layers.
        """
        Middleroad_FC = nn.Sequential(
            nn.Linear(input_size, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
        )
        return Middleroad_FC

    def Highroad_Middleroad_FC(self, input_size):
        """
        :param input_size: The output of the highroad is 4096 and middleroad is 512. Therefore, the input size is
        determined as 4608)
        :return: the Sequential Layers for the last fully connected network of the model.
        """
        Highroad_Middleroad_FC = nn.Sequential(
            nn.Linear(input_size, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        return Highroad_Middleroad_FC

    def forward(self, x):
        """
        :param x: receives the mini batch input with dimension of (N, C, H, W)
        :return: returns the raw output of VCA model. the intermediate output of each layer has the label of "y_" in
        front of layer name.
        """

        # saving the input prior to the computation
        input_image = x

        # computed output of the highroad
        y_Highroad_output = self.VCA_Highroad(input_image)

        # computed output of early layers in VGG-16 model to create input for bottom-up attention module.
        y_Middleroad = self.VGG_Middleroad(input_image)

        # compute each part of bottom-up attention module
        y_Middleroad_MaxPool = self.Middleroad_MaxPool(y_Middleroad)
        y_Global_MaxPool = self.Global_MaxPool(y_Middleroad)
        y_MaxPool1 = self.MaxPool1(y_Middleroad)
        y_MaxPool2 = self.MaxPool2(y_Middleroad)
        y_MaxPool3 = self.MaxPool3(y_Middleroad)

        y_MaxPool1_Global_Adaptive_Pool2 = self.Global_AdaptivePool2(y_MaxPool1)
        y_MaxPool2_Global_Adaptive_Pool2 = self.Global_AdaptivePool2(y_MaxPool2)
        y_MaxPool3_Global_Adaptive_Pool2 = self.Global_AdaptivePool2(y_MaxPool3)

        eca_module_input = torch.cat([y_Global_MaxPool,
                                      y_MaxPool1_Global_Adaptive_Pool2,
                                      y_MaxPool2_Global_Adaptive_Pool2,
                                      y_MaxPool3_Global_Adaptive_Pool2], dim=1)

        # flatten the tensors to concatenate them
        y_eca_module_output_Global_Adaptive_Pool1 = self.Global_AdaptivePool1(eca_module_input)

        # change the dimensionality of the 2 tensors before concatenation
        y_eca_module_output_Global_Adaptive_Pool1 = y_eca_module_output_Global_Adaptive_Pool1.view(
            y_eca_module_output_Global_Adaptive_Pool1.size(0), -1)
        y_Middleroad_MaxPool = y_Middleroad_MaxPool.view(y_Middleroad_MaxPool.size(0), -1)

        # concatenate the computed features from the attention module to feed the fully connected network in middleroad.
        y_Bottom_Up_Attention_Feature = torch.cat([y_Middleroad_MaxPool, y_eca_module_output_Global_Adaptive_Pool1], dim=1)

        # fully connected network of the middleroad.
        y_Middleroad_output = self.VCA_Middleroad(y_Bottom_Up_Attention_Feature)

        # concatenate the 1 dimensional feature tensor from high road and middle road.
        Highroad_Middleroad_input = torch.cat([y_Highroad_output, y_Middleroad_output], dim=1)

        # last fully connected network to compute the high dimensional features based on the middle road and high road
        # features.
        y_Highroad_Middleroad_FC = self.VCA_FC(Highroad_Middleroad_input)

        return y_Highroad_Middleroad_FC


