import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.manifold import TSNE
import seaborn as sns

r"""
<Manifold_Visualization>

This code performs the T distributed stochastic neighbor embedding (t-SNE) of feature vectors extracted from the Visual 
Cortex Amygdala Model. The perplexity variable was fixed to 1182, because this showed the clear manifold of artificial 
neurons in terms of valence decoding. 

Prior to running this code, you must run the Channel_Activity_Extraction.py in both "image" and "gabor" mode to make
inputs for this code. After running "Channel_Activity_Extraction.py", you can run this code to make figures of t-SNE.

This code scans through every files in "result_dir", and look for the file which contains the "model_name" in the file 
name. If you ran the "Channel_Activity_Extraction.py" properly, you must have the "image" and "gabor" version of csv 
file. Then, choose the "layer_idx" to specify which layer activation you want to perform the t-SNE analysis. 

Args:
    result_dir: Different with other codes, this variable is used to scan the csv files to perform the analysis. 
    model_name: you need to specify the model name to find the csv file in the result_dir. In the result_dir, you should find the 2 csv files including the model name.
    n_components: number of components for the t-SNE analysis. 2 should be always remain constant, because we are projecting the features on 2-D t-SNE space. 
    layer_idx: which layer you want to extract. For VCA_FC, we have 2 layers except the output layer, which are 1 & 4. 
    
"""

plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Parameters ')

parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction2', type=str,
                    help='where all the results are saved.')
parser.add_argument('--model_name', default='base_model_conditioned_orientation_epoch100', type=str,
                    help='which epoch you want to perform manifold analysis')
parser.add_argument('--module_to_extract', default='VCA_FC', type=str)
parser.add_argument('--n_components', default=2, type=int,
                    help='The dimension of the latent space you want to project your data.')
parser.add_argument('--layer_idx', default=4, type=int)

args = parser.parse_args()
all_files = os.listdir(args.result_dir)

target_csv = []
for file1 in all_files:
    if args.model_name in file1:
        target_csv.append(file1)
    else:
        pass

#target_csv = all_files

image_csvs = []

for img_file in target_csv:
    if args.module_to_extract in img_file:
        if 'image' in img_file:
            image_csvs.append(img_file)

gabor_csvs = []
for mod_file in target_csv:
    if args.module_to_extract in mod_file:
        if 'gabor' in mod_file:
            gabor_csvs.append(mod_file)

# making labels for the visual labels for the gabors.

X_gabor = pd.DataFrame()

for gabor_csv in gabor_csvs:

    # Gabor Dataframe Processing
    gabor_df = pd.read_csv(os.path.join(args.result_dir, gabor_csv))
    gabor_sub_df = gabor_df[gabor_df['layer_idx'] == args.layer_idx]
    gabor_sub_df = gabor_sub_df.dropna(axis='columns')
    gabor_sub_df = gabor_sub_df[~gabor_sub_df['image_name'].str.contains('cont0')]
    gabor_sub_df['image_name'] = gabor_sub_df['image_name'].str.replace('.png', '')
    gabor_sub_df[['gabor', 'envelope', 'orientation', 'frequency', 'contrast']] = gabor_sub_df['image_name'].apply(
        lambda x: pd.Series(str(x).split("-")))
    gabor_sub_df['frequency'] = gabor_sub_df['frequency'].str.replace('freq', '')
    gabor_sub_df['contrast'] = gabor_sub_df['contrast'].str.replace('cont', '')

    gabor_sub_df['orientation'] = gabor_sub_df['orientation'].astype(int)
    gabor_sub_df['frequency'] = gabor_sub_df['frequency'].astype(int)
    gabor_sub_df['contrast'] = gabor_sub_df['contrast'].astype(int)

    if 'conditioned' not in gabor_csv:
        epoch = 0
    elif 'conditioned' in gabor_csv:
        split = gabor_csv.split('_')
        epoch = split[4]
        epoch = int(epoch.replace('epoch', ''))
    gabor_sub_df['epoch'] = epoch

    # Image Dataframe Processing

    X_gabor = X_gabor.append(gabor_sub_df, ignore_index=True)

X_image = pd.DataFrame()

for image_csv in image_csvs:
    # Gabor Dataframe Processing
    image_df = pd.read_csv(os.path.join(args.result_dir, image_csv))
    image_sub_df = image_df[image_df['layer_idx'] == args.layer_idx]
    image_sub_df = image_sub_df.dropna(axis='columns')
    image_sub_df['image_name'] = image_sub_df['image_name'].str.replace('.png', '')

    if 'conditioned' not in image_csv:
        epoch = 0

    elif 'conditioned' in image_csv:
        split = image_csv.split('_')
        epoch = split[4]
        epoch = int(epoch.replace('epoch', ''))
    image_sub_df['epoch'] = epoch

    X_image = X_image.append(image_sub_df, ignore_index=True)


# gabor patch dataset
#X_gabor = X_gabor.loc[X_gabor['epoch'].isin([0])]

X_gabor.pop('gabor')
X_gabor.pop('envelope')
orientation = X_gabor.pop('orientation')
frequency = X_gabor.pop('frequency')
contrast = X_gabor.pop('contrast')
epoch_gabor = X_gabor.pop('epoch')
X_gabor = X_gabor.iloc[:, 4:]

# image dataset
#X_image = X_image.loc[X_image['epoch'].isin([0])]
epoch_image = X_image.pop('epoch')
valence = X_image.pop('valence')
X_image = X_image.iloc[:, 3:]

gabor_n = np.shape(X_gabor)[0]
image_n = np.shape(X_image)[0]

# Whole dataset
X = X_gabor.append(X_image, ignore_index=True)

tsne = TSNE(n_components=args.n_components, perplexity=10.0, verbose=1, n_iter=1000, random_state=0)
transformed_X = tsne.fit_transform(X)

X_gabor_transformed = transformed_X[0:gabor_n, :]
X_image_transformed = transformed_X[gabor_n:, :]


fig = plt.figure()
g = fig.add_subplot(1,1,1)

sns.set_context('notebook', font_scale=1.1)
sns.set(style="white", palette="muted")
sns.set(rc={'figure.figsize': (7.27, 5.27)})


g = sns.scatterplot(x=X_image_transformed[:, 0],
                    y=X_image_transformed[:, 1],
                    style=epoch_image,
                    palette=sns.diverging_palette(300, 250, l=50, center="dark", as_cmap=True),
                    hue=valence,
                    s=40,
                    alpha=1.0)

g = sns.scatterplot(x=X_gabor_transformed[:, 0],
                    y=X_gabor_transformed[:, 1],
                    palette=sns.diverging_palette(300, 250, l=65, center="dark", as_cmap=True),
                    hue=orientation,
                    style=epoch_gabor,
                    legend='brief',
                    s=150,
                    alpha=1.0)

plt.title("t-SNE Results", weight='bold').set_fontsize('15')
plt.xlabel("t-SNE axis 1", weight='bold').set_fontsize('10')
plt.ylabel("t-SNE axis 2", weight='bold').set_fontsize('10')
plt.legend(loc=2, bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
new_name = args.module_to_extract + '_' + str(args.layer_idx)
#g.figure.savefig(os.path.join('./result/model_test/', new_name + '.pdf'), dpi=300, format="pdf")
