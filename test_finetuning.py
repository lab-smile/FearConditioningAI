from __future__ import print_function, division
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import reg_eval_model, load_checkpoint
from dataloader import reg_dataloader, cond_dataloader, quadrant_finetune_dataloader
import argparse
from datetime import datetime
import csv

'''
#### the following models are **not including attention in low-road and also not have gist** implemented  ######
if running the provided trained model you may use this network  instead of models.network
 from network import VGGReg, Amygdala, VGGbnReg, Amygdala_lowroad

if you retrained one new model, you should use this 
  from models.network import VGGReg, Amygdala, VGGbnReg, Amygdala_lowroad 
instead; this is because at the beginning ,we did not move the network.py into the models folder;

if you are training a model including attention in the low-road or have gist included, you should use 
one of the following: 
from models.Amygdala_lowroad**.py 
It depends on your model you're training
'''

from models.VGG_Model import VGG_Freeze_conv, VGG_BN_Freeze_conv


from torchsummary import summary
# import time
plt.ion()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='./data/', type=str, help='the data root folder')
parser.add_argument('--TRAIN', default='IAPS_10-10-80_train3', type=str, help='the folder of training data')
parser.add_argument('--VAL', default='IAPS_10-10-80_val3', type=str, help='the folder of validation data')
parser.add_argument('--TEST', default='IAPS_10-10-80_test3', type=str, help='the folder of test data')

parser.add_argument('--csv_train', default='./data/IAPS_10-10-80_train3.csv', type=str, help='the path of training data csv file')
parser.add_argument('--csv_val', default='./data/IAPS_10-10-80_val3.csv', type=str, help='the path of training data csv file')
parser.add_argument('--csv_test', default='./data/IAPS_10-10-80_test3.csv',
                    type=str, help='the path of training data csv file')

parser.add_argument('--batch_size', default=64, type=int, help='batch size')

parser.add_argument('--model_to_run', default=2, type=int, help='which model you want to run with experiment')
parser.add_argument('--model_dir', default='/home/seowung/Brain inspired AI_Fear Conditioning/code_new/savedmodel', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='best_vgg_bn_iaps_finetuned_quadrant_batch64_lr7e-4.pth',
                    type=str, help='name of the trained model；'
                                   '***Note**: You have to select which model can perform best on other datasets such as IAPS rather than the best model tested on the training data'
                                   'Usually, the model that has the best generalizability is the trained model obtained at earlier epochs such as 3rd epoch.')


parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--csv_result_recording_folder', default='result/model_test/', type=str, help='the folder including the results csv')
parser.add_argument('--csv_result_recording_name', default='model_regression_test_results.csv')

import glob


def iter_run_allmodels(dir, modelName):
    path = os.path.join(dir,modelName[:-4]+'*.pth')
    for file_name in glob.iglob(path, recursive=True):
        print(file_name)

if __name__ == '__main__':

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = os.path.join(args.model_dir,args.model_name[:-4]+'*.pth')
    file_list = list(glob.iglob(path, recursive=True))
    if len(file_list)<1:
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

            for file_name in sorted(file_list):
                print('\n Test model:{}'.format(file_name))

                test_datasets = args.TEST.split(',')
                test_datasets_labels = args.csv_test.split(',')
                num_datasets = len(test_datasets)
                for i in range(num_datasets):
                    dataset = test_datasets[i]
                    label = test_datasets_labels[i]
                    print('\n Test data:{}'.format(dataset))

                    path = os.path.join(args.model_dir, args.model_name[:-4] + '*.pth')

                    if args.model_to_run == 1:
                        model = VGG_Freeze_conv()
                    elif args.model_to_run == 2:
                        model = VGG_BN_Freeze_conv()

                    dataloaders, dataset_sizes = quadrant_finetune_dataloader(args.data_dir, args.TRAIN, args.VAL, dataset,
                                                                args.csv_train, args.csv_val, label,
                                                                args.batch_size, istrain=False)
                    # Load the  model
                    model = load_checkpoint(model, file_name)
                    # print(model)
                    model = model.to(device)
                    # summary(model, (3, 224, 224))

                    criterion = nn.MSELoss()

                    # ----------------test-----------------------------#
                    score_c_r = reg_eval_model(dataloaders, dataset_sizes, dataset, model, criterion, device)
                    writer.writerow([file_name, dataset, score_c_r])
                    # ----------------Visualize -----------------------------
                    # reg_visualize_model(dataloaders, model, args.TEST, 6, device)