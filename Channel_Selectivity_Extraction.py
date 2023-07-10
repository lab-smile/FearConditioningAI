r"""
This code is used to extract the Channel-Wise Selectivity across the layers in 4 modules of VCA. The Modules
include VGG, Highroad Classifier (a.k.a Fully Connected Layers), Lowroad Classifier, and Final Fully Connected Layers.
If you want to extract the tuning curve of Efficient Channel-Wise Attention Module, Please use ECA_Module_Selectivity
Extraction.

    Args:
         -gabor_dir: This is where the gabor patches are saved for the tuning curve extraction.

         -result_dir: This is where the result file of this code will be saved in.

         --gabor_to_extract: The actual gabor patches you want to use from the gabor_dir can be specified here.
         \The input is form of a list.

         --model_dir: The directory where the trained models are saved.

         --model_name: The name of the model you want to extract the tuning curve.

         --module_to_extract: The name of the module you would like to extract. This prevents you from extracting all
         \layers from the model.

"""

from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataloader import image_patch_loader, Quadrant_Processing
import argparse

matplotlib.use('Agg')

# import time
plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--gabor_dir', default='./data/gabor_extract2', type=str, help='the data root folder')
parser.add_argument('--result_dir', default='./result/model_test', type=str)
parser.add_argument('--gabor_to_extract', action='store', dest='gabor_to_extract', type=str, nargs='*',
                    default=['gabor-gaussian-0-freq20-cont50.png',
                             'gabor-gaussian-15-freq20-cont50.png',
                             'gabor-gaussian-30-freq20-cont50.png',
                             'gabor-gaussian-45-freq20-cont50.png',
                             'gabor-gaussian-60-freq20-cont50.png',
                             'gabor-gaussian-75-freq20-cont50.png',
                             'gabor-gaussian-90-freq20-cont50.png',
                             'gabor-gaussian-105-freq20-cont50.png',
                             'gabor-gaussian-120-freq20-cont50.png',
                             'gabor-gaussian-135-freq20-cont50.png',
                             'gabor-gaussian-150-freq20-cont50.png',
                             'gabor-gaussian-165-freq20-cont50.png',
                             'gabor-gaussian-180-freq20-cont50.png'],
                    help="This part is usually for placing one CS+"
                         "Examples: -i item1 item2, -i item3")

parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='base_model_vca_IAPS_quadrant.pth',
                    type=str, help='name of the trained model；')
parser.add_argument('--module_to_extract', default='VCA_FC', type=str,
                    help='There are 5 different modules to extract. '

                         '\nVCA_Features : This part is the feature extractor part from the VGG16 '

                         '\nVCA_Highroad_Classifier : This part is the classifier part from the VGG16'

                         '\nVCA_Lowroad_Classifier : This part is the fully connected layers from the bottom-up '
                         'attention module, which performs non-linear computation after the attention output.'

                         '\nVCA_FC : This part is the fully connected layers which aggregates the output from the '
                         'Highroad and Lowroad.')

# Params
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

result_dir = os.path.join(args.result_dir, 'Channel_Selectivity_Extraction', args.module_to_extract)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

file_output = pd.DataFrame(
    columns=['model_name', 'module_to_extract', 'layer_idx', 'filter_idx', 'gabor_patch', 'activation'])
file_list = os.listdir(args.model_dir)

for file_name in file_list:
    if file_name != args.model_name:
        print('\n The model {} is not the file you want to run the analysis'.format(file_name))
        continue

    elif file_name == args.model_name:
        print('\n >>>> Model to extract:{}'.format(file_name))

        trained_model = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cuda:0')
        model = trained_model['model']
        model.load_state_dict(trained_model['state_dict'])

        print(model)

        if args.module_to_extract == 'VCA_Features':
            module = model.VCA_Highroad.features
        elif args.module_to_extract == 'VCA_Highroad_Classifier':
            module = model.VCA_Highroad.classifier
        elif args.module_to_extract == 'VCA_Lowroad_Classifier':
            module = model.VCA_Lowroad
        elif args.module_to_extract == 'VCA_FC':
            module = model.VCA_FC
        else:
            raise ValueError

        print('\n >>>> Module to extract : {}'.format(args.module_to_extract))
        print(module)

        n_layer = len(module)
        print('\n Number of layers {}'.format(n_layer))

        image_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        gabor_patch_transform = transforms.Compose([transforms.Resize(size=(image_size, image_size)),
                                                    transforms.CenterCrop(image_size),
                                                    Quadrant_Processing(2),
                                                    transforms.ToTensor(),
                                                    normalize])
        for gabor_patch in args.gabor_to_extract:
            name = args.model_name[:-4]
            orientation = int(gabor_patch.split('-')[2])
            print('\n **** Extracting the activation of gabor patch {} ****'.format(gabor_patch))
            gabor_path = os.path.join(args.gabor_dir, gabor_patch)
            inputs = image_patch_loader(gabor_patch_transform, gabor_path)

            model = model.to(device)
            inputs = inputs.to(device)
            activation = {}

            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()

                return hook


            for n in range(n_layer):
                if isinstance(module[n], nn.ReLU):
                    module[n].register_forward_hook(get_activation('ReLU'+str(n)))
                    outputs = model(inputs)
                    vector = activation['ReLU'+str(n)].cpu().detach().numpy()
                    if vector.ndim == 4:
                        vector = np.mean(np.squeeze(vector), axis=(1, 2))
                    elif vector.ndim == 2:
                        vector = np.squeeze(vector)
                else:
                    continue

                d = {}

                for filter in range(len(vector)):
                    d['model_name'] = [args.model_name]
                    d['module_to_extract'] = [args.module_to_extract]
                    d['layer_idx'] = [n]
                    d['filter_idx'] = [filter + 1]
                    d['gabor_patch'] = [gabor_patch]
                    d['orientation'] = [orientation]
                    d['activation'] = [vector[filter]]
                    filter_output = pd.DataFrame(data=d, columns=['model_name',
                                                                  'module_to_extract',
                                                                  'layer_idx',
                                                                  'filter_idx',
                                                                  'gabor_patch',
                                                                  'orientation',
                                                                  'activation'])
                    file_output = pd.concat([file_output, filter_output], ignore_index=True)
        file_output.to_csv(os.path.join(result_dir, file_name[:-4] + '.csv'))
