# Braininspired AI: Fear Conditioning - v001

*modified Mar.06.2023* \
*contributor : Seowung Leem*

This project aims to develop deep learning architecture based computational model to test the fear conditioning scenario and gain insights in neuroscience perspective. 


## Steps to run the analysis

### Datasets

Make sure the datasets are all in the `./data` directory. If you don't have the data in right format, you will eventually have an error during the training and the analysis.



### Code
1. train_regression.py
- Trains the model from scratch. In the actual work, we first trained the model with ckvideoframe dataset, and then finetuned the model with International Affective Picture System (IAPS) dataset. How to train and finetune the model can be found in **Contents** section. 

2. train_finetuning.py
- Tunes the model into the quadrant input dimension. The input dataset should remain the same with the dataset from the finetuning step from *train_regression.py*. You can play around with the hyperparameters to reach the local minimum. 

3. train_conditioning2.py
- Condition the model. The validation, and test input does not influence the conditioning procedure. However, you have to either make the files remain as default, or revise them to prevent the error from the dataloader. How to conditiong the model can be found in **Contents** section. 

4. test_gaborpatch_iteration.py
5. test_gaborpatches.py


## Contents

***(They are still in process, so each files have dates on the back of its name. The file and the readme will be modified anytime when there is an update.)*** 


### train_regression.py

This code is written to develop the model from scratch. Below are the variables that you can define to customize your training. 

**Args:**
- data_dir: this is the root folder for your data. Fill out the directory where your data is stored.
- TRAIN: the name of your directory where the training set is stored.
- VAL: the name of your directory where the validation set is stored.
- TEST: the name of your directory where the test set is stored.
- csv_train: the name of csv file which contains the image name, type, and valence for training set. 
- csv_val: the name of csv file which contains the image name, type, and valence for validation set. 
- csv_test: the name of csv file which contains the image name, type, and valence for test set. 
- batch_size: the size of single batch you want to feed your network per iteration.
- epoch: number of epoch you want to train your model.
-lr: learning rate.
- model_to_run: the int type code of which model you want to run the training.
- model_dir: the name of your directory where the result model is saved. 
- model_name: the name of your trained model. 
- resume: whether you will resume the training procedure (this works only in terminal)
- start_epoch: which epoch you will start training (if it is resumed, the epoch will not be 1)
- is_fine_tune: is your model training from scratch? or fine tuning some model which is already trained
- file_name: name of the fine tuned model. The fine tuned model should be located in the same directory of 'model_dir'



**example:**
```
python train_regression.py --data_dir ./data --TRAIN IAPS_split_train --VAL IAPS_split_val --TEST IAPS_split_test --csv_train IAPS_split_train.csv --csv_val IAPS_split_val.csv --csv_test IAPS_split_test.csv --batch_size 10 --lr 1e-4 --model_to_run 6 --model_dir ./savedmodel --is_fine_tune True 
```

*Because of the confidentiality issue, the IAPS dataset will not be provided to the public.*

### train_finetuning.py

This code is written to finetune the model into quadrant input dimension. The input model of this code is the final output model from the train_regression.py. 

**Args:**
- data_dir: this is the root folder for your data. Fill out the directory where your data is stored.
- TRAIN: the name of your directory where the training set is stored.
- VAL: the name of your directory where the validation set is stored.
- TEST: the name of your directory where the test set is stored.
- csv_train: the name of csv file which contains the image name, type, and valence for training set. 
- csv_val: the name of csv file which contains the image name, type, and valence for validation set. 
- csv_test: the name of csv file which contains the image name, type, and valence for test set. 
- batch_size: the size of single batch you want to feed your network per iteration.
- epoch: number of epoch you want to train your model.
-lr: learning rate.
- model_to_run: the int type code of which model you want to run the training.
- model_dir: the name of your directory where the result model is saved. 
- model_name: the name of your trained model. 
- resume: whether you will resume the training procedure (this works only in terminal)
- start_epoch: which epoch you will start training (if it is resumed, the epoch will not be 1)
- is_fine_tune: is your model training from scratch? or fine tuning some model which is already trained
- file_name: name of the fine tuned model. The fine tuned model should be located in the same directory of 'model_dir'



**example:**
```
python train_finetuning.py --data_dir ./data --TRAIN IAPS_split_train --VAL IAPS_split_val --TEST IAPS_split_test --csv_train IAPS_split_train.csv --csv_val IAPS_split_val.csv --csv_test IAPS_split_test.csv --batch_size 10 --lr 1e-4 --model_to_run 6 --model_dir ./savedmodel --is_fine_tune True 
```
