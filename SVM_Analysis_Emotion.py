import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

r"""
<SVM_Analysis Emotion>

Trains an RBF-kernel SVM on network feature representations of the
Unconditioned Stimulus (US) images, using each image's valence to assign a
binary pleasant/unpleasant label. The classifier is then evaluated in one of
two modes, controlled by the `what_to_analyze` variable below:

    'US' - held-out US images (standard train/test split)
    'CS' - Gabor patch conditioned stimuli, to test whether the valence
           learned from US images generalizes to the orientations used
           during conditioning (30/45/60 deg -> unpleasant,
           120/135/150 deg -> pleasant)

The train/test split is repeated 100 times with different random seeds and
the resulting accuracy scores are summarized (mean/std) and written to a
single CSV in `result_dir/SVM_Analysis_Emotion/`.

Args:
    result_dir: Directory containing the per-model/per-layer feature CSVs
        produced by the feature-extraction step; also where this script's
        output is written.
    model_dir: Directory containing the saved model checkpoints.
    initial_model: Name of the pre-conditioning model (unused directly here,
        kept for parity with related analysis scripts).
    conditioned_model: Name of the post-conditioning model whose feature CSVs
        will be analyzed.
    module_name: Name of the network module whose activations were extracted.
    layer_idx: Index of the layer within `module_name` to analyze.
"""

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction', type=str,
                    help='where all the results are saved.')
parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where all the results are saved')
parser.add_argument('--initial_model', default='base_model_vca_IAPS_quadrant', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--conditioned_model', default='base_model_conditioned_orientation_epoch100', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--module_name', default='VCA_FC', type=str)
parser.add_argument('--layer_idx', default=4, type=int)
parser.add_argument('--what_to_analyze', default='US', type=str, help='Choose either "US" or "CS" as input.')

# import arguments and load all files
args = parser.parse_args()
all_files = os.listdir(args.result_dir)  # Reading all the files from the result_dir
model_list = os.listdir(args.model_dir)  # Read names of all the trained models from the model_dir
model_list_r = [args.conditioned_model]  # get rid of the extension from the model's name
file_output = pd.DataFrame()  # predefining the output

clf_rbf = make_pipeline(StandardScaler(),
                           SVC(kernel='rbf', gamma='auto'))

# Iterate over models
for model_name in model_list_r:
    result = {}

    target_csv = []
    for file in all_files:
        if model_name in file and args.module_name in file:
            target_csv.append(file)
        else:
            pass

for csv in target_csv:  # call the csvs of image and gabor in separate dataframe.
    if 'image' in csv:
        image_df = pd.read_csv(os.path.join(args.result_dir, csv))
    elif 'gabor' in csv:
        gabor_df = pd.read_csv(os.path.join(args.result_dir, csv))

image_sub_df = image_df[image_df['layer_idx'] == args.layer_idx].dropna(axis='columns')
gabor_sub_df = gabor_df[gabor_df['layer_idx'] == args.layer_idx]
gabor_sub_df = gabor_sub_df.dropna(axis='columns')

gabor_sub_df['label'] = 0
gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-45-')), 'label'] = 0
gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-135-')), 'label'] = 1
gabor_sub_df=gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-45-|-135-'))]

image_sub_df['label'] = 0
image_sub_df.loc[image_sub_df['valence'] <= 4.3, 'label'] = 0
image_sub_df.loc[image_sub_df['valence'] > 6.0, 'label'] = 1
image_label = image_sub_df.pop('label')

what_to_analyze = args.what_to_analyze

if what_to_analyze == 'US':
    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(image_sub_df, image_label, test_size=0.5, random_state=i,
                                                                stratify=image_label)

        X_train = X_train.iloc[:, 4:]
        X_test = X_test.iloc[:, 4:]

        clf_rbf.fit(X_train, y_train)
        prediction = clf_rbf.predict(X_test)
        score = clf_rbf.score(X_test, y_test)
        result['model_name'] = [model_name]
        result['iteration'] = [i+1]
        result['module_name'] = [args.module_name]
        result['score'] = [score]

        filter_output = pd.DataFrame(data=result, columns=['model_name',
                                                           'module_name',
                                                           'score'])
        file_output = pd.concat([file_output, filter_output], ignore_index=True)

    print('\nTested Model: {}'.format(model_name))
    print('Mean accuracy over {} iterations: {:.4f} (+/- {:.4f})'.format(
        len(file_output), file_output['score'].mean(), file_output['score'].std()))

    output_dir = os.path.join(args.result_dir, 'SVM_Analysis_Emotion')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if what_to_analyze == 'CS':
    y_test = gabor_sub_df.pop('label')
    for i in range(100):

        X_train, _, y_train, _ = train_test_split(image_sub_df, image_label, test_size=0.5, random_state=i,
                                                                stratify=image_label)
        X_test = gabor_sub_df.iloc[:, 4:]
        X_train = X_train.iloc[:, 4:]

        clf_rbf.fit(X_train, y_train)
        prediction = clf_rbf.predict(X_test)
        score = clf_rbf.score(X_test, y_test)
        result['model_name'] = [model_name]
        result['iteration'] = [i+1]
        result['module_name'] = [args.module_name]
        result['score'] = [score]

        filter_output = pd.DataFrame(data=result, columns=['model_name',
                                                           'module_name',
                                                           'score'])
        file_output = pd.concat([file_output, filter_output], ignore_index=True)

    print('\nTested Model: {}'.format(model_name))
    print('Mean accuracy over {} iterations: {:.4f} (+/- {:.4f})'.format(
        len(file_output), file_output['score'].mean(), file_output['score'].std()))
    output_dir = os.path.join(args.result_dir, 'SVM_Analysis_Emotion')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

file_output.to_csv(os.path.join(output_dir, args.module_name + '_' + str(args.layer_idx)+'_' + str(what_to_analyze)+'.csv'))
