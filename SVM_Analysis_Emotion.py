import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction2', type=str,
                    help='where all the results are saved.')
parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where all the results are saved')
parser.add_argument('--initial_model', default='base_model_vca_IAPS_quadrant', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--conditioned_model', default='base_model_conditioned_orientation_epoch100', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--module_name', default='VCA_Lowroad_Classifier', type=str)
parser.add_argument('--layer_idx', default=1, type=int)

r"""
<SVM_Analysis Emotion>

This code performs the rbf (gaussian) kernel SVM classification. The model is trained on the feature representations
of Unconditional stimulus in the network. Then, they are tested on the CS+ and other orientations which has the 
generalized valence through the CS+. In the work, 30, 45, 60 was labeled as unpleasant, and 120, 135 and 150 was labeled
as pleasant. 

Args:
    result_dir: Different with other codes, this variable is used to scan the csv files to perform the analysis. 
    initial_model: you need to specify the model name to find the csv file in the result_dir. This is set as the initial model before conditioning. 
    conditioned_model: specify the name of the conditioned model. 
"""

args = parser.parse_args()
all_files = os.listdir(args.result_dir)  # Reading all the files from the result_dir
model_list = os.listdir(args.model_dir)  # Read names of all the trained models from the model_dir
model_list_r = [args.initial_model, args.conditioned_model]  # get rid of the extension from the model's name
file_output = pd.DataFrame()  # predefining the output

clf_rbf = make_pipeline(StandardScaler(),
                           SVC(kernel='rbf', gamma='auto'))

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
    gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-45-|-30-|-60-')), 'label'] = 0
    gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-135-|-120-|-150-')), 'label'] = 1
    gabor_sub_df=gabor_sub_df.loc[(gabor_sub_df['image_name'].str.contains('-45-|-30-|-60-|-135-|-120-|-150-'))]

    image_sub_df['label'] = 0
    image_sub_df.loc[image_sub_df['valence'] <= 4.3, 'label'] = 0
    image_sub_df.loc[image_sub_df['valence'] > 6.0, 'label'] = 1

    y_image = image_sub_df.pop('label')
    y_test = gabor_sub_df.pop('label')

    X_image = image_sub_df.iloc[:, 4:]
    X_test = gabor_sub_df.iloc[:, 4:]

    for i in range(100):
        X_train, X_val, y_train, y_val = train_test_split(X_image, y_image, test_size=0.2, random_state=i,
                                                      stratify=y_image)

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

        print('\n Tested Model: {}'.format(model_name))
        print("%0.2f accuracy" % (score))

output_dir = os.path.join(args.result_dir, 'SVM_Analysis_Emotion')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_output.to_csv(os.path.join(output_dir, args.module_name + '_' + str(args.layer_idx)+'.csv'))
