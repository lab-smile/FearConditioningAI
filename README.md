# Braininspired AI: Fear Conditioning - v001

*modified Mar.06.2023* \
*contributor : Seowung Leem*

This project aims to develop deep learning architecture based computational model to test the fear conditioning scenario and gain insights in neuroscience perspective. 
## Contents

In the Fear Conditioning repo, there are currently 2 codes. These codes are codes for running the each step of preprocessing, training and test phase of Neuroscience-AI: Fear Conditioning project. 

***(They are still in process, so each files have dates on the back of its name. The file and the readme will be modified anytime when there is an update.)*** 

1) IAPS_Data_preprocessing
2) Alex_Net_Pretrained

### IAPS_Data_preprocessing.py

This code is written to load the dataset. The dataset used in current code is IAPS dataset. Code imports the dataset from the directory, resize the image into 224 x 224 2D array, and turn them into Pytorch tensor. Depending on the input, the function imports the image from `unpleasant` images, or `neutral` images

```
img, emotion = import_image(unpleasant_directory, unpleasant_xlsx_directory, emotion='unpleasant')
```

*Because of the confidentiality issue, the IAPS dataset will not be provided to the public.*

### Alex_Net_Pretrained.py

This code defines the model and run the training based on the model. The base architecture is Image-Net pretrained AlexNet with additional ANN layers for 2 class classification. The training is done by k-fold cross validation and the value of `k` can be modified as the user wants. Currently, the model and training function is not seperated. In later version, the codes will be modularized. 
