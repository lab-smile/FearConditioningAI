import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

r"""
<SVM_Analysis Before_After>

This code performs the rbf (gaussian) kernel SVM classification. The model is trained on the gabor feature
representations from before conditioning (initial_model). It is then tested on the gabor representations both
before and after conditioning (initial_model and conditioned_model), to see how well a classifier fit to the
pre-conditioning representation generalizes to the post-conditioning representation. In the work, 30, 45, 60 was
labeled as unpleasant, and 120, 135 and 150 was labeled as pleasant.

Args:
    result_dir: Different with other codes, this variable is used to scan the csv files to perform the analysis.
    initial_model: you need to specify the model name to find the csv file in the result_dir. This is set as the initial model before conditioning.
    conditioned_model: specify the name of the conditioned model.
"""

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction', type=str,
                    help='where all the results are saved.')
parser.add_argument('--model_dir', default='./savedmodel/', type=str, help='where all the results are saved')
parser.add_argument('--initial_model', default='base_model_vca_IAPS_quadrant', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--conditioned_model', default='base_model_conditioned_orientation_epoch100', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--module_name', default='VCA_FC', type=str)
parser.add_argument('--layer_idx', default=1, type=int)


def SVM_Data(gabor_df):
    """Filter one layer's Gabor-patch activation CSV down to the 45deg/135deg orientations
    used as CS+ during conditioning, and split it into (features, binary label) for SVM use.

    Label 0 = 45deg (paired with unpleasant US), label 1 = 135deg (paired with pleasant US).
    """
    gabor_sub_df = gabor_df[gabor_df['layer_idx'] == args.layer_idx]
    gabor_sub_df = gabor_sub_df.dropna(axis='columns')
    gabor_sub_df = gabor_sub_df.loc[~gabor_sub_df['image_name'].str.contains('cont0\.')]

    gabor_sub_df['label'] = 0
    gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-45-')), 'label'] = 0
    gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-135-')), 'label'] = 1
    gabor_sub_df = gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-45-|-135-'))]

    y_test = gabor_sub_df.pop('label')
    X_test = gabor_sub_df.iloc[:, 4:]

    return X_test, y_test

args = parser.parse_args()
all_files = os.listdir(args.result_dir)  # Reading all the files from the result_dir
model_list = os.listdir(args.model_dir)  # Read names of all the trained models from the model_dir
model_list_r = [args.initial_model, args.conditioned_model]  # get rid of the extension from the model's name
file_output = pd.DataFrame()  # predefining the output

# Loading feature representation of CS+ and US
target_csv = []
for file in all_files:
    if args.initial_model in file and args.module_name in file:
        target_csv.append(file)
    else:
        pass

for csv in target_csv:  # call the gabor csv into a dataframe.
    if 'gabor' in csv:
        gabor_df = pd.read_csv(os.path.join(args.result_dir, csv))

X_test_before, y_test_before = SVM_Data(gabor_df)


target_csv = []
for file in all_files:
    if args.conditioned_model in file and args.module_name in file:
        target_csv.append(file)
    else:
        pass

for csv in target_csv:  # call the gabor csv into a dataframe.
    if 'gabor' in csv:
        gabor_df = pd.read_csv(os.path.join(args.result_dir, csv))

X_test_after, y_test_after = SVM_Data(gabor_df)

result = {}

for i in range(100):
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', random_state=i))
    clf.fit(X_test_before, y_test_before)

    score_before = clf.score(X_test_before, y_test_before)
    score_after = clf.score(X_test_after, y_test_after)

    result['module_name'] = [args.module_name]
    result['layer_idx'] = [args.layer_idx]
    result['before gabor accuracy'] = [score_before]
    result['after gabor accuracy'] = [score_after]

    filter_output = pd.DataFrame(data=result, columns=['module_name',
                                                       'layer_idx',
                                                       'before gabor accuracy',
                                                       'after gabor accuracy'
                                                       ])
    file_output = pd.concat([file_output, filter_output], ignore_index=True)

print('Mean accuracy over {} runs:'.format(len(file_output)))
print('  before gabor accuracy: {}'.format(file_output['before gabor accuracy'].mean()))
print('  after gabor accuracy: {}'.format(file_output['after gabor accuracy'].mean()))

output_dir = os.path.join(args.result_dir, 'SVM_Analysis_Before_After')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_output.to_csv(os.path.join(output_dir, args.module_name + '_' + str(args.layer_idx)+'.csv'))
