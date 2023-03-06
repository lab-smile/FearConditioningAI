from __future__ import print_function, division
import glob
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import reg_eval_model, load_checkpoint
from dataloader import gabor_patch_loader, Quadrant_Processing
import argparse
from datetime import datetime
import csv

from models.network import VGGReg, Amygdala, VGGbnReg, Amygdala_lowroad
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from models.Amygdala_lowroad_attention_Ishighroadonly_gist_model6v2 import \
    Amygdala_lowroad as Amygdala_lowroad_gist_model6v2

# import time
plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='./data/gabor_patch_full/freq20', type=str, help='the data root folder')
parser.add_argument('--gabor_patch', default='gabor-gaussian-45-freq20-cont50.png', type=str, help='the folder of test data')

parser.add_argument('--model_dir',
                    default='./savedmodel/Conditioning_Experiment12',
                    type=str, help='where to save the trained model')

parser.add_argument('--initial_model_name',
                    default='Amy_IntermediateRoadAttention_lr1e5_stepsize30_bs16_ep100_FinetuningIAPS_10-10-80_3_Mar082022_epoch62_quadrant_epoch34.pth',
                    type=str, help='name of the trained model；'
                                   '***Note**: You have to select which model can perform best on other datasets such as IAPS rather than the best model tested on the training data'
                                   'Usually, the model that has the best generalizability is the trained model obtained at earlier epochs such as 3rd epoch.')

parser.add_argument('--conditioned_model_name',
                    default='Amy_IntermediateRoadAttention_lr1e5_stepsize30_bs16_ep100_FinetuningIAPS_10-10-80_3_Mar082022_epoch62_quadrant_epoch34_conditioned_man90_epoch100.pth',
                    type=str, help='name of the trained model；'
                                   '***Note**: You have to select which model can perform best on other datasets such as IAPS rather than the best model tested on the training data'
                                   'Usually, the model that has the best generalizability is the trained model obtained at earlier epochs such as 3rd epoch.')

parser.add_argument('--model_to_run', default=2, type=int,
                    help=
                    '0: Original VGG, with replacing the last neuron from 1000 to 1'
                    '1: VGG with Amygdala module'
                    '2: VGG with Amygdala module and intermediate pathway of attention module'
                    '3: VGG with Amygdala module, intermediate pathway and low pathway with gist features'
                    )
parser.add_argument('--n_layers', default=3, type=int,
                    help='the number of layers in Amygdala; this setting is only for model_to_run of 1')
parser.add_argument('--lowfea_VGGlayer', default=10, type=int,
                    help='which layer of vgg to extract features as the low-road input. This setting is only available for model_to_run 2')
parser.add_argument('--highfea_VGGlayer', default=36, type=int,
                    help='which layer of vgg to extract features as the high-road input to amygdala. '
                         'This setting is only available for model_to_run 2')

parser.add_argument('--is_highroad_only', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether only applying high-road for test; the low-road information will be assigned as zeros')
parser.add_argument('--is_gist', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply gist as input to low-road instead')

parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--csv_result_recording_folder', default='result/model_test/', type=str,
                    help='the folder including the results csv')
parser.add_argument('--csv_result_recording_name', default='model_regression_test_results.csv')

'''
This code is used to determine if the model is conditioned or not by using gabor patch. 
'''


def iter_run_allmodels(dir, modelName):
    path = os.path.join(dir, modelName[:-4] + '*.pth')
    for file_name in glob.iglob(path, recursive=True):
        print(file_name)


if __name__ == '__main__':

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    initial_path = os.path.join(args.model_dir, args.initial_model_name[:-4] + '.pth')
    conditioned_path = os.path.join(args.model_dir, args.conditioned_model_name[:-4] + '.pth')
    file_list = [initial_path, conditioned_path]
    is_gist = args.is_gist
    is_saliency = False
    if len(file_list) < 1:
        print('Cannot find models!')
    else:
        if not os.path.exists(args.csv_result_recording_folder):
            os.makedirs(args.csv_result_recording_folder)
        path_result_csv = os.path.join(args.csv_result_recording_folder, args.csv_result_recording_name)
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        path_result_csv = path_result_csv[:-4] + '_' + str(now) + '.csv'
        schema = ['model', 'dataset', 'R']
        with open(path_result_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Gives the header name row into csv
            writer.writerow([g for g in schema])

            initial_pred = 0
            conditioned_pred = 0

            for file_name in sorted(file_list):
                print('\n Test model:{}'.format(file_name))

                if args.model_to_run == 0:
                    model = VGGReg()
                elif args.model_to_run == 1:
                    model = Amygdala(args.n_layers)
                elif args.model_to_run == 2:
                    model = Amy_IntermediateRoad(lowfea_VGGlayer=args.lowfea_VGGlayer,
                                                 highfea_VGGlayer=args.highfea_VGGlayer)
                    # model = Amygdala_lowroad_attention(lowfea_VGGlayer=args.lowfea_VGGlayer,
                    # highfea_VGGlayer=args.highfea_VGGlayer)
                else:
                    is_gist = True
                    model = Amygdala_lowroad_gist_model6v2(lowfea_VGGlayer=args.lowfea_VGGlayer,
                                                           highfea_VGGlayer=args.highfea_VGGlayer,
                                                           is_gist=is_gist)

                    # dataloaders, dataset_sizes = reg_dataloader(args.data_dir, args.TRAIN, args.VAL, dataset,
                    #                                            args.csv_train, args.csv_val, label,
                    #                                            args.batch_size, istrain=False, is_gist= is_gist, is_saliency=is_saliency)

                image_size = 224
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

                gabor_patch_transform = transforms.Compose([
                    transforms.Resize(size=(image_size, image_size)),
                    transforms.CenterCrop(image_size),
                    Quadrant_Processing(2),
                    transforms.ToTensor(),
                    normalize])

                gabor_path = os.path.join(args.data_dir, args.gabor_patch)

                inputs = gabor_patch_loader(gabor_patch_transform, gabor_path)

                # Load the  model
                model = load_checkpoint(model, file_name)
                # print(model)
                model = model.to(device)
                inputs = inputs.to(device)
                # summary(model, (3, 224, 224))

                criterion = nn.MSELoss()

                outputs = model(inputs)

                outputs = 1 + outputs * (9 - 1)  # normalize to 1-9   Because our network last layer is sigmoid,
                # which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]

                preds = outputs.reshape(torch.Size([1])).cpu().detach().numpy()

                if file_name == initial_path:
                    initial_pred = preds
                else:
                    conditioned_pred = preds

            # ----------------test-----------------------------#

            print("The gabor patch used in test:", args.gabor_patch)
            print("The initial valence: {:.4f}".format(initial_pred[0]))
            print('Valence after conditioning : {:.4f}'.format(conditioned_pred[0]))
            # ----------------Visualize -----------------------------
            # reg_visualize_model(dataloaders, model, args.TEST, 6, device)
