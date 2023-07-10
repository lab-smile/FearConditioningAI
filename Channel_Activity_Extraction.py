# base libraries
import os
import argparse

# libraries for arithmetic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# libraries for pytorch
import torch
import torch.nn as nn
from torchvision import transforms

# project based libraries
from dataloader import Quadrant_Processing, image_patch_loader


r"""
<Channel_Activity_Extraction>

This code is used to extract the activity of corresponding images from the model. In specific, this code is used to 
extract the activation of fully connected layers in different pathways. 

The output of this code is used to "Distance_Analysis", "Manifold Visualization".
2 different analysis require following for each module. "image_to_extract", "data_dir", "image_dir", "gabor_dir" should
be exclusively defined

1. For Distance Analysis, use "gabor" for -image_to_extract- argument. No "image" required. 
- for "gabor_dir", use directory with conditional stimulus.

2. For Manifold Visualization, use "gabor" for -image_to_extract- argument. 
- for "gabor_dir", use directory with gabor patches with different orientation. 
- for "image_dir", use directory with unconditional stimulus. 
"""

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--image_to_extract', default='image', type=str, help='Choose either "image" or "gabor" as input.')
parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction')

parser.add_argument('--data_dir', default='./data', type=str, help='where all the data are saved.')

parser.add_argument('--image_dir', default='IAPS_US', type=str,
                    help='where the unpleasant data are saved.')

parser.add_argument('--gabor_dir', default='gabor_extract2', type=str,
                    help='where the gabor patch you want to extract is saved.')

parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where the model is saved')

parser.add_argument('--module_to_extract', default='VCA_FC', type=str,
                    help='There are 4 different modules to extract. '
                         '\nVCA_Features : This part is the feature extractor part from the VGG16 '
                         '\nVCA_Highroad_Classifier : This part is the classifier part from the VGG16'
                         '\nVCA_Middleroad_Classifier : This part is the fully connected layers from the bottom-up '
                         'attention module, which performs non-linear computation after the attention output.'
                         '\nVCA_FC : This part is the fully connected layers which aggregates the output from the '
                         'Highroad and Middleroad.')

# Params
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# if there's no result dir
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# image size should be fixed as default as well as normalization parameters
image_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# defining which image you want to extract the activation and its transformation
if args.image_to_extract == 'gabor':
    image_file = args.gabor_dir

elif args.image_to_extract == 'image':
    image_file = args.image_dir

if args.image_to_extract == 'gabor':
    image_transform = transforms.Compose([transforms.Resize(size=(image_size, image_size)),
                                          transforms.CenterCrop(image_size),
                                          Quadrant_Processing(2),
                                          transforms.ToTensor(),
                                          normalize])
elif args.image_to_extract == 'image':
    image_transform = transforms.Compose([transforms.Resize(size=(image_size, image_size)),
                                          transforms.CenterCrop(image_size),
                                          Quadrant_Processing(4),
                                          transforms.ToTensor(),
                                          normalize])

# initializing model list & image file list
model_list = []
images = []

for model in os.listdir(args.model_dir):
    if not os.path.isdir(model):
        if not model.startswith('.'):
            model_list.append(model)

image_file_path = os.path.join(args.data_dir, image_file)
for image in os.listdir(image_file_path):
    if not os.path.isdir(image):
        if not image.startswith('.'):
            images.append(image)

# loop over the models to extract the activation
for model_name in model_list:
    file_output = pd.DataFrame()

    # import the model and trained weights
    trained_model = torch.load(os.path.join(args.model_dir, model_name), map_location='cuda:0')
    model = trained_model['model']
    model.load_state_dict(trained_model['state_dict'])

    # define the module for extraction
    if args.module_to_extract == 'VCA_Features':
        module = model.VCA_Highroad.features
    elif args.module_to_extract == 'VCA_Highroad_Classifier':
        module = model.VCA_Highroad.classifier
    elif args.module_to_extract == 'VCA_Middleroad_Classifier':
        module = model.VCA_Middleroad
    elif args.module_to_extract == 'VCA_FC':
        module = model.VCA_FC
    elif args.module_to_extract == 'ECA_Module':
        module = model.ECA_Module
    else:
        raise ValueError

    print('>>>> Model to extract:{}'.format(model_name))
    print('>>>> Module to extract:{}'.format(args.module_to_extract))
    print('>>>> Extract Mode:{}'.format(args.image_to_extract))

    # if extracting the activation for image, save original & predicted valence.
    if args.image_to_extract == 'image':
        valence_df = pd.read_csv(os.path.join(args.data_dir, image_file+'.csv'), names=['image', 'extension', 'sd', 'valence'], dtype={'image':"string"})
        valence_df['image'] = valence_df['image'].astype(str)

    # loop over the images
    for image in images:
        image_path = os.path.join(image_file_path, image)
        inputs = image_patch_loader(image_transform, image_path)

        # define the layer index as 0 for ECA module.
        if args.module_to_extract != 'ECA_Module':
            n_layer = len(module)
        elif args.module_to_extract == 'ECA_Module':
            n_layer = 0

        activation = {}
        model = model.to(device)
        inputs = inputs.to(device)

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # extraction if the target module is not ECA
        if args.module_to_extract != 'ECA_Module':
            for n in range(n_layer):
                # hook the relu layers
                if isinstance(module[n], nn.ReLU):
                    module[n].register_forward_hook(get_activation('ReLU'+str(n)))
                # hook the sigmoid layers (depreciated)
                elif isinstance(module[n], nn.Sigmoid):
                    module[n].register_forward_hook(get_activation('Sigmoid'+str(n)))
                else:
                    continue
                outputs = model(inputs)
                outputs = 1 + outputs * (9 - 1)
                outputs = outputs.cpu().detach().numpy()

                # checking the valence for the sanity check
                print('The valence of the input:{}'.format(outputs[0][0]), flush=True)

                if isinstance(module[n], nn.ReLU):
                    vector = activation['ReLU' + str(n)].cpu().detach().numpy()
                else:
                    pass

                # different squeeze format for conv layers and fully connected layers
                if vector.ndim == 4:
                    vector = np.mean(np.squeeze(vector), axis=(1, 2))
                elif vector.ndim == 2:
                    vector = np.squeeze(vector)

                d = {}
                d['image_name'] = [image]
                d['layer_idx'] = [n]

                # dataframe of image file and module
                if args.image_to_extract == 'gabor':
                    d['valence'] = outputs[0]
                elif args.image_to_extract == 'image':
                    d['valence'] = [valence_df[valence_df['image'] == image[:-4]]['valence'].iloc[0]]

                # dataframe of activation values
                vector_df = pd.DataFrame(vector).T

                # make it as a one row dataframe
                filter_output = pd.DataFrame(data=d)
                filter_output = filter_output.join(vector_df)

                # concatenate 2 dataframes
                file_output = pd.concat([file_output, filter_output], ignore_index=True)

        # extraction if the target module is ECA
        elif args.module_to_extract == 'ECA_Module':
            # define the keys for every layers in the module
            module.avg_pool.register_forward_hook(get_activation('AdaptiveAvgPool2d'))
            module.sigmoid.register_forward_hook(get_activation('Sigmoid'))
            module.conv.register_forward_hook(get_activation('Conv1d'))
            module.register_forward_hook(get_activation('Output'))

            # feed-forward
            outputs = model(inputs)
            outputs = 1 + outputs * (9 - 1)
            outputs = outputs.cpu().detach().numpy()

            # iterate over each layer and get the extracted value.
            for layer, _ in activation.items():
                vector = activation[layer].cpu().detach().numpy()

                # different squeeze format for conv layers and fully connected layers
                if vector.ndim == 4:
                    vector = np.mean(np.squeeze(vector, axis=(0,)), axis=(1, 2))
                elif vector.ndim == 3:
                    vector = np.squeeze(vector)
                elif vector.ndim == 2:
                    vector = np.squeeze(vector)

                d = {}
                d['image_name'] = [image]
                d['layer_idx'] = [layer]

                # dataframe of image file and module
                if args.image_to_extract == 'gabor':
                    d['valence'] = [outputs[0]]
                elif args.image_to_extract == 'image':
                    d['valence'] = [valence_df[valence_df['image'] == image[:-4]]['valence'].iloc[0]]

                # dataframe of activation values
                vector_df = pd.DataFrame(vector).T

                # make it as a one row dataframe
                filter_output = pd.DataFrame(data=d)
                filter_output = filter_output.join(vector_df)
                file_output = pd.concat([file_output, filter_output], ignore_index=True)

    # save the result
    file_output.to_csv(os.path.join(args.result_dir, model_name[:-4] + '_' + args.image_to_extract + '_'
                                    + args.module_to_extract + '.csv'))



