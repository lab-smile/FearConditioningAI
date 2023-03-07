from __future__ import print_function, division
import glob

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import load_checkpoint
from dataloader import image_patch_loader, Quadrant_Processing
import argparse
from datetime import datetime
import csv

from models.VGG_Model import VGG,VGG_Freeze_conv, VGG_BN_Freeze_conv, VGG_Freeze_conv_FC1, VGG_Freeze_conv_FC2, Visual_Cortex_Amygdala, Visual_Cortex_Amygdala_wo_Attention, VGG_Freeze_conv_FC2_attention, VGG_Freeze_conv_attention
from models.VGG_Model_Conditioning import VGG_Freeze_conv1_FC2_attention
# import time
plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Params
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--data_dir', default='./data/gabor_patch_full/freq20', type=str, help='the data root folder')
parser.add_argument('--gabor_patch', default='gabor-gaussian-135-freq20-cont50.png', type=str, help='the folder of test data')

parser.add_argument('--model_dir',
                    default='/home/seowung/Brain inspired AI_Fear Conditioning/code_new/savedmodel/',
                    type=str, help='where to save the trained model')

parser.add_argument('--model_to_run', default=6, type=int, help='which model you want to run with experiment')

parser.add_argument('--initial_model_name',
                    default='base_model_vca_IAPS_quadrant.pth',
                    type=str, help='name of the trained model；'
                                   '***Note**: You have to select which model can perform best on other datasets such as IAPS rather than the best model tested on the training data'
                                   'Usually, the model that has the best generalizability is the trained model obtained at earlier epochs such as 3rd epoch.')

parser.add_argument('--conditioned_model_name',
                    default='base_model_conditioned_orientation_epoch100.pth',
                    type=str, help='name of the trained model；'
                                   '***Note**: You have to select which model can perform best on other datasets such as IAPS rather than the best model tested on the training data'
                                   'Usually, the model that has the best generalizability is the trained model obtained at earlier epochs such as 3rd epoch.')


parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--csv_result_recording_folder', default='./result/model_test/', type=str,
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

    result_df = pd.DataFrame(columns=['model', 'pred'])
    file_list = []

    for file in os.listdir(args.model_dir):
        if not os.path.isdir(file):
            if not file.startswith('.'):
                file_list.append(file)

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

                file_path = os.path.join(args.model_dir, file_name)

                if args.model_to_run == 1:
                    model = VGG()
                elif args.model_to_run == 2:
                    model = VGG_Freeze_conv()
                elif args.model_to_run == 3:
                    model = VGG_BN_Freeze_conv()
                elif args.model_to_run == 4:
                    model = VGG_Freeze_conv_FC1()
                elif args.model_to_run == 5:
                    model = VGG_Freeze_conv_FC2()
                elif args.model_to_run == 6:
                    model = Visual_Cortex_Amygdala()
                elif args.model_to_run == 7:
                    model = Visual_Cortex_Amygdala_wo_Attention()
                elif args.model_to_run == 8:
                    model = VGG_Freeze_conv_FC2_attention()
                elif args.model_to_run == 9:
                    model = VGG_Freeze_conv1_FC2_attention

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

                inputs = image_patch_loader(gabor_patch_transform, gabor_path)

                # Load the  model
                model = load_checkpoint(model, file_path)
                # print(model)
                model = model.to(device)
                inputs = inputs.to(device)
                # summary(model, (3, 224, 224))

                criterion = nn.MSELoss()

                outputs = model(inputs)

                outputs = 1 + outputs * (9 - 1)  # normalize to 1-9   Because our network last layer is sigmoid,
                # which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]

                preds = outputs.reshape(torch.Size([1])).cpu().detach().numpy()

                if file_name == args.initial_model_name:
                    initial_pred = preds
                    sub_df = pd.DataFrame([[file_name, initial_pred[0]]], columns=['model', 'pred'])
                else:
                    conditioned_pred = preds
                    sub_df = pd.DataFrame([[file_name, conditioned_pred[0]]], columns=['model', 'pred'])

                result_df = result_df.append(sub_df, ignore_index=True)
                result_df = result_df.sort_values(by=['model'], ascending=True)

            # ----------------test-----------------------------#

            print("The gabor patch used in test:", args.gabor_patch)
            print("The initial valence: {:.4f}".format(initial_pred[0]))
            print('Valence after conditioning : {:.4f}'.format(conditioned_pred[0]))
            result_df.to_csv(args.csv_result_recording_folder+'Conditioning_Experiment ' + args.gabor_patch[:-4] + '.csv')
            # ----------------Visualize -----------------------------
            # reg_visualize_model(dataloaders, model, args.TEST, 6, device)
