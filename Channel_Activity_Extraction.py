import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataloader import Quadrant_Processing, image_patch_loader
import argparse

r"""
<Channel_Activity_Extraction>

This code is used to extract the activity of corresponding images from the model. In specific, this code is used to 
extract the activation of last 2 fully connected layers of VCA model. 

The output of this code is used to "Manifold Analysis".
"""

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--image_to_extract', default='gabor', type=str, help='Choose either "image" or "gabor" as input.')
parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction')

parser.add_argument('--data_dir', default='./data', type=str, help='where all the data are saved.')

parser.add_argument('--image_dir1', default=['IAPS_all'], type=str,
                    help='where the unpleasant data are saved.')
parser.add_argument('--csv_dir1', default='IAPS_all.csv', type=str)

parser.add_argument('--image_dir2', default='IAPS_Conditioning_Pleasant2', type=str,
                    help='where the pleasant data are saved.')
parser.add_argument('--csv_dir2', default='IAPS_Conditioning_Pleasant2.csv', type=str)

parser.add_argument('--gabor_dir', default=['gabor_extract2'], type=list,
                    help='where the gabor patch you want to extract is saved.')

parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where the model is saved')

parser.add_argument('--module_to_extract', default='VCA_Lowroad_Classifier', type=str,
                    help='There are 4 different modules to extract. '
                         '\nVCA_Features : This part is the feature extractor part from the VGG16 '
                         '\nVCA_Highroad_Classifier : This part is the classifier part from the VGG16'
                         '\nVCA_Lowroad_Classifier : This part is the fully connected layers from the bottom-up '
                         'attention module, which performs non-linear computation after the attention output.'
                         '\nVCA_FC : This part is the fully connected layers which aggregates the output from the '
                         'Highroad and Lowroad.')

# Params
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if there's no result dir
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

image_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if args.image_to_extract == 'gabor':
    image_file = args.gabor_dir

elif args.image_to_extract == 'image':
    #image_file = [args.image_dir1, args.image_dir2]
    image_file = args.image_dir1

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

model_list = os.listdir(args.model_dir)

for model_name in model_list:

    file_output = pd.DataFrame()

    trained_model = torch.load(os.path.join(args.model_dir, model_name), map_location='cuda:0')
    model = trained_model['model']
    model.load_state_dict(trained_model['state_dict'])

    if args.module_to_extract == 'VCA_Features':
        module = model.VCA_Highroad.features
    elif args.module_to_extract == 'VCA_Highroad_Classifier':
        module = model.VCA_Highroad.classifier
    elif args.module_to_extract == 'VCA_Lowroad_Classifier':
        module = model.VCA_Lowroad
    elif args.module_to_extract == 'VCA_FC':
        module = model.VCA_FC
    elif args.module_to_extract == 'ECA_Module':
        module = model.ECA_Module
    else:
        raise ValueError

    print('\n >>>> Model to extract:{}'.format(model_name))
    print('>>>> Module to extract:{}'.format(args.module_to_extract))
    print('>>>> Extract Mode:{}'.format(args.image_to_extract))

    for file in image_file:
        image_file_path = os.path.join(args.data_dir, file)
        images = os.listdir(image_file_path)
        if args.image_to_extract == 'image':
            valence_df = pd.read_csv(os.path.join(args.data_dir, file+'.csv'), names=['image', 'extension', 'sd', 'valence'], dtype={'image':"string"})
            valence_df['image'] = valence_df['image'].astype(str)
        for image in images:
            image_path = os.path.join(image_file_path, image)
            inputs = image_patch_loader(image_transform, image_path)

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

            if args.module_to_extract != 'ECA_Module':
                for n in range(n_layer):
                    if isinstance(module[n], nn.ReLU):
                        module[n].register_forward_hook(get_activation('ReLU'+str(n)))
                    elif isinstance(module[n], nn.Sigmoid):
                        module[n].register_forward_hook(get_activation('Sigmoid'+str(n)))
                    else:
                        continue
                    outputs = model(inputs)
                    outputs = 1 + outputs * (9 - 1)
                    outputs = outputs.cpu().detach().numpy()

                    if isinstance(module[n], nn.ReLU):
                        vector = activation['ReLU' + str(n)].cpu().detach().numpy()
                    else:
                        pass

                    if vector.ndim == 4:
                        vector = np.mean(np.squeeze(vector), axis=(1, 2))
                    elif vector.ndim == 2:
                        vector = np.squeeze(vector)

                    d = {}
                    d['image_name'] = [image]
                    d['layer_idx'] = [n]

                    if args.image_to_extract == 'gabor':
                        d['valence'] = [outputs[0]]
                    elif args.image_to_extract == 'image':
                        d['valence'] = [valence_df[valence_df['image'] == image[:-4]]['valence'].iloc[0]]

                    vector_df = pd.DataFrame(vector).T

                    filter_output = pd.DataFrame(data=d)
                    filter_output = filter_output.join(vector_df)

                    file_output = pd.concat([file_output, filter_output], ignore_index=True)

            elif args.module_to_extract == 'ECA_Module':
                module.avg_pool.register_forward_hook(get_activation('AdaptiveAvgPool2d'))
                module.sigmoid.register_forward_hook(get_activation('Sigmoid'))
                module.conv.register_forward_hook(get_activation('Conv1d'))
                module.register_forward_hook(get_activation('Output'))
                outputs = model(inputs)
                outputs = 1 + outputs * (9 - 1)
                outputs = outputs.cpu().detach().numpy()

                for layer, _ in activation.items():
                    vector = activation[layer].cpu().detach().numpy()

                    if vector.ndim == 4:
                        vector = np.mean(np.squeeze(vector, axis=(0,)), axis=(1, 2))
                    elif vector.ndim == 3:
                        vector = np.squeeze(vector)
                    elif vector.ndim == 2:
                        vector = np.squeeze(vector)

                    d = {}
                    d['image_name'] = [image]
                    d['layer_idx'] = [layer]

                    if args.image_to_extract == 'gabor':
                        d['valence'] = [outputs[0]]
                    elif args.image_to_extract == 'image':
                        d['valence'] = [valence_df[valence_df['image'] == image[:-4]]['valence'].iloc[0]]

                    vector_df = pd.DataFrame(vector).T

                    filter_output = pd.DataFrame(data=d)
                    filter_output = filter_output.join(vector_df)

                    file_output = pd.concat([file_output, filter_output], ignore_index=True)

    file_output.to_csv(os.path.join(args.result_dir, model_name[:-4] + '_' + args.image_to_extract + '_'
                                    + args.module_to_extract + '.csv'))



