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
parser.add_argument('--result_file_header', default='model_test', type=str,
                    help='the name of the result folder header you want to concatenate')
parser.add_argument('-l', '--result_columns', action='store', dest='result_columns', type=str, nargs='*',
                    default=['model', 'gabor', 'valence'], help="Examples: -i item1 item2, -i item3")
parser.add_argument('-r', '--columns_to_extract', action='store', dest='columns_to_extract', type=str, nargs='*',
                    default=['freq0', 'freq2', 'freq4', 'freq6', 'freq8', 'freq10',
                             'freq12', 'freq14', 'freq16', 'freq18', 'freq20',
                             'freq22', 'freq24', 'freq26', 'freq28', 'freq30',
                             'freq32', 'freq34', 'freq36', 'freq38', 'freq40'],
                    help="Examples: -i item1 item2, -i item3")
parser.add_argument('-n', '--models_to_extract', dest='models_to_extract', type=str,
                    default='base_model_conditioned_orientation_epoch100.pth',
                    help="Examples: -i item1 item2, -i item3")
parser.add_argument('--output_csv', default='Processed_File_3D.csv', type=str,
                    help='the name of the result folder header you want to concatenate')

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
    for freq in args.columns_to_extract:
        avg_df = pd.DataFrame()
        for j, test_file in enumerate(test_files_all):
            csv_dir = os.path.join(args.result_dir, test_file)
            csv_files_all = []
            for csv_files in os.listdir(csv_dir):
                if csv_files.startswith(args.result_file_header):

                    csv_file_path = os.path.join(csv_dir, csv_files)  # file directory
                    df = pd.read_csv(csv_file_path, header=0)  # read the csv files by pandas

                    freq_df = df[df['model'] == args.models_to_extract]  # sample the model df
                    freq_col = freq_df[
                        ['model', 'gabor', freq]].copy()  # extract the Gabors and the corresponding frequency data.

                    if avg_df.empty:
                        column_names = df['gabor'].str.split(pat='-', expand=True)
                        avg_df['model'] = freq_col['model']
                        avg_df['orientation'] = column_names[2]
                        avg_df['frequency'] = freq
                        avg_df['contrast'] = column_names[4].str.replace('.png', '')
                        avg_df[test_file] = freq_col[freq]

                    elif not avg_df.empty:
                        avg_df[test_file] = freq_col[freq]
        output_name = freq + '_' + args.output_csv
        avg_df.to_csv(os.path.join(args.result_dir, output_name))
