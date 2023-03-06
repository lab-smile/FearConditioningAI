from __future__ import print_function, division
import glob
import torch
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
parser.add_argument('--result_dir', default='./result/', type=str, help='the location where the results are saved')
parser.add_argument('--result_file_header', default='Conditioning_Experiment_gabor-gaussian', type=str, help='the name of the result folder header you want to concatenate')
parser.add_argument('--result_columns', default=['model', 'gabor', 'valence'], help='the name of the result folder header you want to concatenate')
parser.add_argument('-l', '--test_gabor', action='store', dest='test_gabor', type=str, nargs='*',
                    default=['gabor-gaussian-90-freq20-cont20', 'gabor-gaussian-90-freq20-cont80'], help="Examples: -i item1 item2, -i item3")

'''
This code is for extracting n different gabor patches to make csv file for learning curve. The result_dir should have 
the directory name starting with 'model_test'. Then it concatenates all the gabor data in the model_test files. 
The output file is the concatenation of 10 iterations for same experiments. 

The file which gives the input to this function: test_gaborpatch_iteration.py

'''

if __name__ == '__main__':

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_files_all = []
    dataframe_un = pd.DataFrame(columns=['model'])
    dataframe_pl = pd.DataFrame(columns=['model'])

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
            model = df['model']
            pred = df['pred']

            if args.test_gabor[0] in csv_files_all[i] and '135' not in csv_files_all[i]:
                dataframe_un['model'] = model
                dataframe_un['test'+str(j+1)] = pred

            if args.test_gabor[1] in csv_files_all[i]:
                dataframe_pl['model'] = model
                dataframe_pl['test' + str(j+1)] = pred

    dataframe_un.to_csv(args.result_dir + '/Learning_Curve_'+str(args.test_gabor[0])+'.csv')
    dataframe_pl.to_csv(args.result_dir + '/Learning_Curve_'+str(args.test_gabor[1])+'.csv')

