import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.manifold import TSNE

r"""
<Manifold Analysis>

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

parser.add_argument('--result_dir', default='./result/model_test/Channel_Activity_Extraction', type=str, help='where all the results are saved.')
parser.add_argument('--model_name', default='base_model_conditioned_orientation_epoch100', type=str, help='which epoch you want to perform manifold analysis')
parser.add_argument('--n_components', default=2, type=int, help='The dimension of the latent space you want to project your data.')
parser.add_argument('--layer_idx', default=4, type=int)


args = parser.parse_args()
all_files = os.listdir(args.result_dir)

target_csv = []
for file1 in all_files:
    if args.model_name in file1:
        target_csv.append(file1)
    else:
        pass

for file2 in target_csv:
    if 'image' in file2:
        image_csv = file2
    elif 'gabor' in file2:
        gabor_csv = file2


image_df = pd.read_csv(os.path.join(args.result_dir, image_csv))
gabor_df = pd.read_csv(os.path.join(args.result_dir, gabor_csv))

image_sub_df = image_df[image_df['layer_idx'] == args.layer_idx]
gabor_sub_df = gabor_df[gabor_df['layer_idx'] == args.layer_idx]

image_sub_df = image_sub_df.dropna(axis = 'columns')
gabor_sub_df = gabor_sub_df.dropna(axis = 'columns')

x_image = image_sub_df.iloc[:, 4:]
x_gabor = gabor_sub_df.iloc[:, 4:]

image_valence = image_sub_df.iloc[:, 3].to_numpy()
gabor_valence = gabor_sub_df.iloc[:, 3].to_numpy()

x_image = x_image.to_numpy()
x_gabor = x_gabor.to_numpy()

X = np.append(x_image, x_gabor, axis=0)

#mds = MDS(n_components=args.n_components)
#transformed_X = mds.fit_transform(X)

tsne = TSNE(n_components=args.n_components, perplexity=1182.0, verbose=1, n_iter=1000, random_state=0)
transformed_X = tsne.fit_transform(X)

transformed_X_image = transformed_X[:len(image_valence), :]
transformed_X_gabor = transformed_X[-len(gabor_valence):, :]


fig, ax = plt.subplots()
sc = ax.scatter(transformed_X_image[:, 0], transformed_X_image[:, 1], c=image_valence, s=25, alpha=0.8,cmap='viridis')
ax.scatter(transformed_X_gabor[0][0], transformed_X_gabor[0][1], s=100, c='violet', marker='v')
ax.scatter(transformed_X_gabor[1][0], transformed_X_gabor[1][1], s=100, c='cyan', marker='^')
ax.grid(True)
plt.colorbar(sc)  # show color scale
plt.show()





