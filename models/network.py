r"""Legacy/experimental model definitions predating the final Visual_Cortex_Amygdala
architecture in models/VGG_Model.py.

Provides the base VGG-16 implementation (VGG, make_layers, vgg16_ori, vgg16_bn_ori) that
models/VGG_Model.py, models/VGG_Model_Conditioning.py, and models/VGG_classification.py
all import their `vgg16_ori` backbone from (each of those modules also re-defines its
own copy of this same VGG scaffolding; only vgg16_ori/vgg16_bn_ori here are actually used
by other modules).

The remaining classes (VGGReg, VGGbnReg, Amygdala, Amygdala_lowroad, FeatureExtractor,
VggPartial_high) are earlier iterations of the low-road/high-road ("amygdala") design
that were superseded by Visual_Cortex_Amygdala, kept here for reference/reproducibility
of earlier experiments rather than active use in the current training scripts.
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.nn import functional as F
import copy
# from torchvision import models
import numbers
import matplotlib
matplotlib.use('Agg')



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

def vgg16_ori(is_freeze = True):
    # Load the pretrained model from pytorch
    model = _vgg16(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model

def vgg16_bn_ori(is_freeze = True):
    # Load the pretrained model from pytorch
    model = _vgg16_bn(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model

class VGGReg(nn.Module):
    """VGG-16 with its final classifier layer replaced by a single sigmoid unit, for valence regression."""

    def __init__(self,):
        super(VGGReg, self).__init__()
        self.vgg = vgg16_ori()
        # Newly created modules have require_grad=True by default
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1),nn.Sigmoid())
    def forward(self, x):
        return self.vgg(x)

class VGGbnReg(nn.Module):
    """Same as VGGReg, but built on the batch-normalized VGG-16 variant."""

    def __init__(self,):
        super(VGGbnReg, self).__init__()
        self.vgg = vgg16_bn_ori()
        # Newly created modules have require_grad=True by default
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1),nn.Sigmoid())


    def forward(self, x):
        return self.vgg(x)

class Amygdala(nn.Module):
    """VGG-16 backbone with a configurable-depth MLP ("amygdala") regression head bolted on.

    Replaces VGG-16's classifier entirely and instead feeds its penultimate-layer
    features through `number_layer` fully-connected (ReLU + dropout) layers down to a
    single output unit.

    Args:
        number_layer (int): total number of FC layers in the amygdala head (including
            the final output layer).
        n_First (int): width of the first hidden layer.
        n_hidden (int): width of subsequent hidden layers.
    """

    def __init__(self, number_layer, n_First = 1000, n_hidden = 1000 ):
        super(Amygdala, self).__init__()
        self.vgg = vgg16_ori()

        #the first layer size for amygdala
        self.n_First = n_First
        # number of neurons in hidden layer for amygdala network
        self.H =n_hidden
        self.D_out = 1 # the number of neuron in the output
        self.n_layer = number_layer #number of layer of amygdala not including vgg

        #the size input to amygdala, which is the output of vgg
        # self.D_in = self.vgg.classifier[6].out_features
        self.D_in = self.vgg.classifier[6].in_features
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        # self.vgg.classifier[6] = nn.Sequential(nn.Linear(self.D_in, self.n_First),nn.ReLU()) #replace the last layer of VGG
        # build fully connected layers for amygdala
        self.linears_list = nn.ModuleList()
        self.linears_list.append(nn.Sequential(nn.Linear(self.D_in, self.n_First),nn.ReLU(inplace=True), nn.Dropout(p=0.5)))
        if self.n_layer > 2:
            self.linears_list.append(nn.Sequential(nn.Linear(self.n_First, self.H), nn.ReLU(inplace=True), nn.Dropout(p=0.5)))
        for i in range(self.n_layer - 3):
            self.linears_list.append(nn.Sequential(nn.Linear(self.H, self.H), nn.ReLU(inplace=True), nn.Dropout(p=0.5)))
        self.linears_list.append(nn.Linear(self.H, self.D_out))

    #amygdala network is composed of vgg and additional fully connected layers
    def forward(self, x):
        y = self.vgg(x)
        for i in range(len(self.linears_list)):
            y = self.linears_list[i](y)
        return y

    #original vgg16




class Amygdala_lowroad(nn.Module):
    """Early low-road/high-road model: a single max-pooled low road (LA) plus a VGG-16
    high road (CE), predicting a single valence score.

    LA ("Lateral nucleus") takes early VGG-16 features, max-pools them down to a fixed
    512-d vector, and passes them through a small FC network. CE ("Central nucleus")
    concatenates this with the high-level VGG-16 features and predicts valence through
    further FC layers ending in a sigmoid. Superseded by Visual_Cortex_Amygdala in
    models/VGG_Model.py, which adds multi-scale pooling and channel attention to the low road.

    Args:
        lowfea_VGGlayer (int): index into vgg.features where the low road branches off.
        highfea_VGGlayer (int): controls how many of VGG-16's classifier layers are kept
            for the high road (see vgg_highfea_part.classifier slicing below).
    """
    def __init__(self, lowfea_VGGlayer =4, highfea_VGGlayer = 36):
        super(Amygdala_lowroad, self).__init__()
        self.vgg = vgg16_ori()

        # settingn which layers features should be extracted
        self.lowfea_VGGlayer = lowfea_VGGlayer
        self.highfea_VGGlayer = highfea_VGGlayer

        # seperate the network VGG to different parts
        self.vgg_lowfea_part = self.vgg.features[:self.lowfea_VGGlayer]
        self.vgg_lowfea_part[self.lowfea_VGGlayer - 1] = nn.MaxPool2d(kernel_size=40, stride=0, padding=0, dilation=1,
                                                                      ceil_mode=False)  # 40 ->512

        size_input_low_road = 512
        self.amygdala_low = self.LA_low_road(size_input_low_road)

        self.vgg_highfea_part = self.vgg
        print(self.vgg_highfea_part)
        self.vgg_highfea_part.classifier = nn.Sequential(
            *list(self.vgg.children())[2][:(self.highfea_VGGlayer - 30 - 7)])


        self.input_size_CE = 4096 + 512

        self.amygdala_CE = self.CE(self.input_size_CE)


    def LA_low_road(self, size_input):
        """FC head (with dropout) that projects the pooled low-road features down to 512-d."""
        amygdala_low = nn.Sequential(
            nn.Linear(size_input, 512), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(512,512), nn.ReLU(inplace=True), nn.Dropout(p=0.5),

        )
        return amygdala_low


    def CE(self, input_size_CE):
        """FC head that combines low-road + high-road features into a single valence score."""
        amygdala_CE = nn.Sequential(
            nn.Linear(input_size_CE, 1024), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        return amygdala_CE


    #amygdala network is composed of vgg and additional fully connected layers
    def forward(self, x):

        y_low = self.vgg_lowfea_part(x)
        y_low =  y_low.view(y_low.size(0), -1)
        y_low = self.amygdala_low(y_low)

        y_high = self.vgg_highfea_part(x)

        flatten_integrated_feature = torch.cat([y_low, y_high], dim=1)

        # CE network
        regression = self.amygdala_CE(flatten_integrated_feature)


        return regression


    # Note: LA_high_road and amygdala_middle below are not called anywhere in __init__
    # or forward(); they are left over from an earlier version of this model's design.

    def LA_high_road(self, size_input):
        """Unused: FC head with the same shape as LA_low_road, intended for a high-road variant."""
        amygdala_high = nn.Sequential(
            nn.Linear(size_input, 512), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5),

        )
        return amygdala_high

    def amygdala_middle(self, size_input):
        """Unused: a small 5-block conv/maxpool stack, intended for a middle-road variant."""
        amygdala_middle = nn.Sequential(nn.Conv2d(size_input, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,ceil_mode=False),
                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                                             )
        return amygdala_middle


class FeatureExtractor(nn.Module):
    """Wraps a trained model to expose the intermediate activation at one named layer/submodule.

    Given a full model and a `submodel_name` identifying which part of it to probe
    ('vgg_ori', 'vgg', 'amygdala_middle', 'amygdala_LA', or 'amygdala_CE'), slices out
    the layers up to and including `layer_name` and reproduces just enough of the
    original forward pass to compute that submodule's output. Used for feature/activity
    extraction and analysis rather than training.

    Args:
        model (nn.Module): a trained model instance (e.g. an Amygdala_lowroad).
        layer_name (int): index of the last layer/module to include in the slice.
        submodel_name (str): which part of `model` to extract from.
    """

    def __init__(self, model, layer_name, submodel_name):
        super(FeatureExtractor, self).__init__()
        self.layer_name = layer_name
        self.submodel_name = submodel_name


        modules = []

        if submodel_name == 'vgg_ori':
            submodule = model
            modules.append(list(list(submodule.children())[0].children())[0][:self.layer_name+1])
            modules = nn.Sequential(*modules)
        elif submodel_name == 'vgg':
            submodule = copy.deepcopy(model.vgg)
            if layer_name < 30:
                modules = submodule.features[:self.layer_name+1]
            else:
                modules = submodule
                if self.layer_name <= 36:
                    modules.classifier = nn.Sequential(
                        *list(modules.children())[2][:(self.layer_name - 30 - 7)])
        elif submodel_name == 'amygdala_middle':

            self.vgg_lowfea_part =  model.vgg_lowfea_part

            self.amygdala_low = model.amygdala_low

            modules = self.amygdala_low[:self.layer_name + 1]
        elif submodel_name == 'amygdala_LA':

            self.vgg_lowfea_part = model.vgg_lowfea_part
            self.amygdala_low = model.amygdala_low

            self.vgg_highfea_part = model.vgg_highfea_part
            self.vgg_highfea_part.classifier = nn.Sequential(
                *list(self.vgg_highfea_part.children())[2][:(-1)])

        elif submodel_name == 'amygdala_CE':

            self.vgg_lowfea_part = model.vgg_lowfea_part
            self.amygdala_low = model.amygdala_low

            self.vgg_highfea_part = model.vgg_highfea_part
            self.vgg_highfea_part.classifier = nn.Sequential(
                *list(self.vgg_highfea_part.children())[2][:(-1)])

            self.amygdala_CE = model.amygdala_CE

            modules = self.amygdala_CE[:self.layer_name + 1]


        self.partial_model = modules

    def forward(self, x):
        if self.submodel_name in ['vgg', 'vgg_ori']:

            output = self.partial_model(x)

        elif self.submodel_name == 'amygdala_middle':

            y_low = self.vgg_lowfea_part(x)
            y_low = y_low.view(y_low.size(0), -1)
            output = self.partial_model(y_low)

        elif self.submodel_name == 'amygdala_LA':

            y_low_VGG = self.vgg_lowfea_part(x)
            y_low_VGG = y_low_VGG.view(y_low_VGG.size(0), -1)
            y_low =  self.amygdala_low(y_low_VGG)

            y_high = self.vgg_highfea_part(x)
            output = torch.cat([y_low, y_high], dim=1)

        elif self.submodel_name == 'amygdala_CE':

            y_low_VGG = self.vgg_lowfea_part(x)
            y_low_VGG = y_low_VGG.view(y_low_VGG.size(0), -1)
            y_low =  self.amygdala_low(y_low_VGG)

            y_high = self.vgg_highfea_part(x)

            flatten_integrated_feature = torch.cat([y_low, y_high], dim=1)

            output = self.partial_model(flatten_integrated_feature)


        return output






class VggPartial_high(nn.Module):
    """Standalone module that runs a fresh VGG-16's features + a prefix of its classifier,
    i.e. just the "high road" slice used by the Amygdala models above, without the low road.

    Args:
        is_freeze (bool): stored but unused; VGG-16 is always initialized via vgg16_ori()
            which handles freezing internally.
        highfea_VGGlayer (int): controls how many of VGG-16's classifier layers are kept
            (see the same slicing convention as Amygdala_lowroad.vgg_highfea_part).
    """

    def __init__(self, is_freeze=True, highfea_VGGlayer = 33):
        super(VggPartial_high, self).__init__()

        self.is_freeze = is_freeze
        self.vgg = vgg16_ori()

        modules = []
        modules.append(list(self.vgg.children())[0])
        modules.append(list(self.vgg.children())[1][:(highfea_VGGlayer-30-7)])

        self.vgg_highfea_part = nn.Sequential(*modules)


    def forward(self, x):
        x = self.vgg_highfea_part(x)
        return x

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
# Note: duplicate of models/filter_module.py's GaussianSmoothing, kept here for this
# module's own historical experiments.
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


