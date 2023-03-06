from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib

matplotlib.use('Agg')
import time
import os
import copy
from utils import reg_eval_model, save_checkpoint, load_checkpoint
from dataloader import reg_dataloader
import argparse
from scipy.stats import pearsonr
import numpy as np
from GPUtil import showUtilization as gpu_usage
from numba import cuda

from models.VGG_Model import VGG, VGG_Freeze_conv, VGG_BN_Freeze_conv, VGG_Freeze_conv_FC1, VGG_Freeze_conv_FC2, Visual_Cortex_Amygdala, Visual_Cortex_Amygdala_wo_Attention, VGG_Freeze_conv_FC2_attention, VGG_Freeze_conv_attention
r"""This code trains the regression model for emotion decoding. In parser.add_argument, you need to fill out the 
    parameters you need for code to work. The difference of this code with train_finetuing & train_conditioning is the 
    input dimension. train_finetuning and train_conditioning has the quadrant input dimension, designed for conditioning.
    However, this part has the input dimension of full size image, with augmentation method defined in the dataloader. 

    Args:
         --data_dir: this is the root folder for your data. Fill out the directory where your data is stored.
         --TRAIN: the name of your directory where the training set is stored.
         --VAL: the name of your directory where the validation set is stored.
         --TEST: the name of your directory where the test set is stored.
         --csv_train: the name of csv file which contains the image name, type, and valence for training set. 
         --csv_val: the name of csv file which contains the image name, type, and valence for validation set. 
         --csv_test: the name of csv file which contains the image name, type, and valence for test set. 
         --batch_size: the size of single batch you want to feed your network per iteration.
         --epoch: number of epoch you want to train your model.
         --lr: learning rate
         --model_to_run: the int type code of which model you want to run the training. 
         --model_dir: the name of your directory where the result model is saved. 
         --model_name: the name of your trained model. 
         --resume: whether you will resume the training procedure (this works only in terminal)
         --start_epoch: which epoch you will start training (if it is resumed, the epoch will not be 1)
         --is_fine_tune: is your model training from scratch? or fine tuning some model which is already trained
         --file_name: name of the fine tuned model. The fine tuned model should be located in the same directory of 'model_dir'
    """

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='./data', type=str, help='the data root folder')
parser.add_argument('--TRAIN', default='IAPS_120_no60_test', type=str, help='the folder of training data')
parser.add_argument('--VAL', default='IAPS_120_no60_val', type=str, help='the folder of validation data')
parser.add_argument('--TEST', default='IAPS_120_no60', type=str, help='the folder of test data')

parser.add_argument('--csv_train', default='./data/IAPS_120_no60_test.csv', type=str,
                    help='the path of training data csv file')
parser.add_argument('--csv_val', default='./data/IAPS_120_no60_val.csv', type=str,
                    help='the path of training data csv file')
parser.add_argument('--csv_test', default='./data/IAPS_120_no60.csv', type=str,
                    help='the path of training data csv file')

parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
# smaller lr is important or the output will be nan
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for SGD')

parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where to save the trained model')
parser.add_argument('--model_to_run', default=6, type=int, help='which model you want to run with experiment')
parser.add_argument('--model_name',
                    default='vca_IAPS_batch128_lr1e-4.pth',
                    type=str, help='name of the trained model')

parser.add_argument('--resume', default=None, type=str, help='the path to checkpoint')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--is_fine_tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply fine tuning to the model')

parser.add_argument('--file_name',
                    default='vca_ckvideo_batch128_lr2e-5_epoch45.pth',
                    type=str, help='name of the trained model')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# This part is needed just in case to remove all the cached data in the GPU
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def train_model(dataloaders, dataset_sizes, TRAIN, VAL, model, criterion, optimizer, scheduler,
                num_epochs, device, start_epoch=0):
    args = parser.parse_args()
    since = time.time()

    # defining the R and parameter variable to save the intermediate results
    best_model_wts = copy.deepcopy(model.state_dict())
    best_R = 0.0
    best_loss = 0

    # for loop to train and validate the model
    for epoch in range(start_epoch, num_epochs):
        logs = {}
        print('\n Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            labels_list = []
            preds_list = []
            running_loss = 0.0

            # Iterate over data.
            for _, inputs, labels in dataloaders[phase]:

                # move the input and label tensors to the predefined device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)

                    # normalize to 1-9   Because our network last layer is sigmoid,
                    # which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
                    outputs = 1 + outputs * (9 - 1)
                    # learning rate has to be smaller than 1e-3 otherwise the output will be nan
                    loss = criterion(outputs.view(labels.size()), labels.float())

                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data
                    labels_list.extend(labels.cpu().numpy())
                    preds = outputs.reshape(labels.shape).cpu().detach().numpy()
                    preds_list.extend(preds)

            if phase == TRAIN:
                scheduler.step()

            # calculation of the loss, Pearson Correlation Coefficient and R2
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_R, _ = pearsonr(labels_list, preds_list)
            epoch_R2 = np.around(epoch_R ** 2, 2)

            print('{} Loss: {:.4f} R: {:.4f} R2: {:.4f}'.format(phase, epoch_loss, epoch_R, epoch_R2))

            if phase == VAL:
                if epoch == 0 or epoch == start_epoch:
                    best_loss = epoch_loss

            if phase == VAL and epoch_loss < best_loss:
                best_R = epoch_R
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

                save_checkpoint(best_R, best_loss, best_epoch, model,  best_model_wts,
                                optimizer, os.path.join(args.model_dir, args.model_name[:-4]+'_epoch' +str(best_epoch)
                                                      +'.pth'))

                print('Best epoch: {:} Best loss: {:.4f} Best R: {:.4f} '.format(
                    best_epoch, best_loss, best_R))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val R: {:4f}'.format(best_R))
    print('Best epoch: {:}'.format(best_epoch))
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    free_gpu_cache()
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create  the result directory to save the training result in the directory
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists('./result'):
        os.makedirs('./result')

    start_epoch = 0

    # if the running mode is resume, the code will import the model with latest epoch and finish the training.
    if args.resume:
        if os.path.isfile(os.path.join(args.model_dir, args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(os.path.join(args.model_dir, args.resume),map_location='cuda:0')
            if "epoch" in checkpoint:
                start_epoch = checkpoint['epoch']
            else:
                start_epoch = args.start_epoch
            model = checkpoint['model']
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # if the mode is fine-tuning mode, the code will import the pretrained model, and load the model parameters
    elif args.is_fine_tune:
        print("=> loading old model '{}'".format(args.file_name))
        old_model = torch.load(os.path.join(args.model_dir, args.file_name), map_location='cuda:0')
        start_epoch = args.start_epoch
        model = old_model['model']
        model.load_state_dict(old_model['state_dict'])

    # if you are training the model from scratch, the code will import the model architecture you defined from the
    # VGG_Model.py, and use that model as a framework.
    else:
        if args.model_to_run == 1:
            model = VGG()
        elif args.model_to_run == 2:
            model = VGG_Freeze_conv()
        elif args.model_to_run == 3:
            model = VGG_BN_Freeze_conv()
        elif args.model_to_run == 4:
            model = VGG_Freeze_conv_FC1()
        elif args.model_to_run == 5:
            model = VGG_Freeze_conv_FC2()
        elif args.model_to_run == 6:
            model = Visual_Cortex_Amygdala()
        elif args.model_to_run == 7:
            model = Visual_Cortex_Amygdala_wo_Attention()
        elif args.model_to_run == 8:
            model = VGG_Freeze_conv_FC2_attention()
        elif args.model_to_run == 9:
            model = VGG_Freeze_conv_attention()

    # Define the loss function and optimization method. If you change the momentum to 1.0, the gradient will explode and
    # give you unexpected results.
    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Defining the learning rate decay.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    print(model)

    # Feed the model to the GPU device.
    model = model.to(device)

    print("Test before training")
    dataloaders_test, dataset_sizes_test = reg_dataloader(args.data_dir, args.TRAIN, args.VAL, args.TEST,
                                                          args.csv_train, args.csv_val, args.csv_test,
                                                          args.batch_size, istrain=False)

    reg_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device)

    # ---------------- Training-----------------------------#
    print('-' * 10)
    print("Preparing training data and it will take a while...")
    dataloaders_train, dataset_sizes_train = reg_dataloader(args.data_dir,
                                                            args.TRAIN, args.VAL, args.TEST, args.csv_train,
                                                            args.csv_val, args.csv_test, args.batch_size, istrain=True)

    model = train_model(dataloaders_train, dataset_sizes_train, args.TRAIN,
                        args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
                        args.epoch, device, start_epoch)

    # ----------------test-----------------------------#
    reg_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device)


if __name__ == '__main__':
    main()
