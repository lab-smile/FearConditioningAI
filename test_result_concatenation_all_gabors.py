from __future__ import print_function, division
import glob
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


# import time
plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--result_dir', default='./result', type=str, help='the location where the results are saved')
parser.add_argument('--result_file_header', default='Conditioning_Experiment_freq', type=str, help='the name of the result folder header you want to concatenate')
parser.add_argument('--result_columns', default=['model', 'gabor', 'valence'], help='the name of the result folder header you want to concatenate')

'''
This code is to make the csv file of all different frequencies. The input is from the 'test_gaborpatch_iteration.py'. 
After running this code, use test_result_3Dprocess to make a csv file to make a 3D plot.
'''

if __name__ == '__main__':

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_files_all = []
    dataframe = pd.DataFrame(columns=['model'])

    for files in os.listdir(args.result_dir):
        if files.startswith('model'):
            test_files_all.append(files)

    print('Number of test files : {:.4f}'.format(len(test_files_all)))

    # read through the test files
    for j, test_file in enumerate(test_files_all):
        csv_dir = os.path.join(args.result_dir, test_file)
        csv_files_all = []
        for csv_files in os.listdir(csv_dir):
            if csv_files.startswith(args.result_file_header):
                csv_files_all.append(csv_files)

        for i in range(len(csv_files_all)):
            csv_file_path = os.path.join(csv_dir, csv_files_all[i])
            df = pd.read_csv(csv_file_path)
            df.sort_values(by=['model'])

            name = csv_files_all[i]
            name = name.replace(args.result_file_header, '')
            name = name.replace('.csv', '')


            dataframe['model'] = df['model']
            dataframe['gabor'] = df['gabor']
            dataframe['freq'+name] = df['valence']

        dataframe.to_csv(csv_dir + '/'+str(test_file)+'.csv')



















