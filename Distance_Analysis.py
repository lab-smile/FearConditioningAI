# base libraries
import os
import argparse

# libraries for arithmetic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction', type=str,
                    help='where all the results are saved.')
parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where all the results are saved')
parser.add_argument('--module_name', default='VCA_FC', type=str)

'''
This code is used to to compute the feature distance between CS+ conditioned with pleasant and unpleasant emotions. 
For that, you must first perform Channel_Activity_Extraction.py, with gabor patch containing CS+. 
'''

args = parser.parse_args()
# Reading all the files from the result_dir
all_files = os.listdir(args.result_dir)

# Code to filter out the meta files
model_list = []
for file in os.listdir(args.model_dir):
    if not os.path.isdir(file):
        if not file.startswith('.'):
            model_list.append(file)

# get rid of the extension from the model's name
model_list_r = [model.replace('.pth', '') for model in model_list]

# predefining the output
file_output = pd.DataFrame()


for idx, model_name in enumerate(model_list_r):
    target_csv = []

    # loop over the files and filter out the csv files with specific model name and module name.
    for file in all_files:
        if model_name in file and args.module_name in file:
            target_csv.append(file)

    # call the csvs of image and gabor in separate dataframe.
    for csv in target_csv:
        if 'image' in csv:
            image_df = pd.read_csv(os.path.join(args.result_dir, csv))
            layers = np.unique(image_df['layer_idx'].values)
        elif 'gabor' in csv:
            gabor_df = pd.read_csv(os.path.join(args.result_dir, csv))
            layers = np.unique(gabor_df['layer_idx'].values)

    for layer in layers:
        gabor_sub_df = gabor_df[gabor_df['layer_idx'] == layer]
        gabor_sub_df = gabor_sub_df[gabor_df['image_name'].str.contains('-45-|-135-')].dropna(axis='columns')

        X_test = gabor_sub_df.iloc[:, 4:]
        Y_test = gabor_sub_df.pop('valence')

        # normalization of feature vectors in terms of feature dimension.
        X_test = normalize(X_test, norm='l2', axis=0)
        distance = euclidean_distances(X_test)

        result = {}

        result['model_name'] = [model_name]
        result['layer'] = layer
        result['distance'] = distance[0][1]

        filter_output = pd.DataFrame(data=result, columns=['model_name', 'layer', 'distance'])
        file_output = pd.concat([file_output, filter_output], ignore_index=True)

file_output_pivot = file_output.pivot(index="model_name", columns="layer", values="distance")
file_output_pivot.to_csv(os.path.join(args.result_dir, 'Feature_Distance' + '_' + str(args.module_name)+'.csv'))
