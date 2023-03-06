from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib

matplotlib.use('Agg')
import time
import random
import os
import copy
from utils import reg_eval_model, save_checkpoint, load_checkpoint, cond_eval_model
from dataloader import cond_dataloader, cond_dataloader2, cond_dataloader3
import argparse
from logger import Logger
from scipy.stats import pearsonr
import numpy as np

from models.VGG_Model import VGG, VGG_Freeze_conv


r"""This code trains the regression model for emotion decoding. In parser.add_argument, you need to fill out the 
    parameters you need for code to work. 

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
         --model_dir: the name of your directory where the result model is saved. 
         --model_name: the name of your trained model. 
         --logfolder: the name of the model you want to put in the log.
         --is_log: whether you will record the data in to the log file.
         --resume: whether you will resume the training procedure (this works only in terminal)
         --start_epoch: which epoch you will start training (if it is resumed, the epoch will not be 1)
         --model_to_run: the base VCA model which you want to train. 0:VGG, 1:Amy, 2:Amy+IR, 3:Amy+IR+LR(Gist)

    Example:
        data_transforms = data_transform(train_folder, val_folder, test_folder)
    """

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='./data', type=str, help='the data root folder')

parser.add_argument('-l', '--gabor_dir', action='store', dest='gabor_dir', type=str, nargs='*',
                    default=['./data/gabor_CS/gabor-gaussian-45-freq20-cont50.png',
                             './data/gabor_CS/gabor-gaussian-135-freq20-cont50.png'
                             ],
                    help="Examples: -i item1 item2, -i item3")

parser.add_argument('--TRAIN', default='IAPS_Conditioning_Unpleasant2', type=str,
                    help='the folder of training data')
parser.add_argument('--TRAIN2', default='IAPS_10-10-80_train3', type=str,
                    help='the folder of training data')

parser.add_argument('--VAL', default='IAPS_10-10-80_val3', type=str, help='the folder of validation data')
parser.add_argument('--TEST', default='IAPS_10-10-80_test3', type=str, help='the folder of test data')

parser.add_argument('--csv_train', default='./data/IAPS_Conditioning_Unpleasant2.csv', type=str,
                    help='the path of training data csv file')
parser.add_argument('--csv_train2', default='./data/IAPS_10-10-80_train3.csv', type=str,
                    help='the path of training data csv file')

parser.add_argument('--csv_val', default='./data/IAPS_10-10-80_val3.csv', type=str,
                    help='the path of training data csv file')
parser.add_argument('--csv_test', default='./data/IAPS_10-10-80_test3.csv', type=str,
                    help='the path of training data csv file')

parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
# smaller lr is important or the output will be nan
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for SGD')

parser.add_argument('--model_dir', default='./savedmodel/Conditioning_Experiment14', type=str, help='where to save the trained model')
parser.add_argument('--model_name',
                    default='base_model.pth',
                    type=str, help='name of the trained model')
parser.add_argument('--logfolder',
                    default='base_model.pth',
                    type=str, help='name of the log')

parser.add_argument('--is_log', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to record log')

parser.add_argument('--resume', default=None, type=str, help='the path to checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--model_to_run', default=2, type=int,
                    help=
                    '0: Original VGG, with replacing the last neuron from 1000 to 1'
                    '1: VGG with Amygdala module'
                    '2: VGG with Amygdala module and intermediate pathway of attention module'
                    '3: VGG with Amygdala module, intermediate pathway and low pathway with gist features'
                    )

parser.add_argument('--n_layers', default=3, type=int,
                    help='the number of layers in Amygdala; this setting is only for model_to_run of 1')

parser.add_argument('--lowfea_VGGlayer', default=10, type=int,
                    help='which layer of vgg to extract features as the low-road input. '
                         'This setting is only available for model_to_run 2 and 4')

parser.add_argument('--highfea_VGGlayer', default=36, type=int,
                    help='which layer of vgg to extract features as the high-road input to amygdala. '
                         'This setting is only available for model_to_run 2 and 4')

parser.add_argument('--is_gist', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply gist as input to low-road instead')

parser.add_argument('--is_saliency', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply saliency as input to low-road instead')

parser.add_argument('--is_fine_tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply fine tuning to the model')

parser.add_argument('--file_name',
                    default='base_model.pth',
                    type=str, help='name of the trained model')

parser.add_argument('--manipulation', default=0.0, type=int, help='Probability to block out the unconditional stimulus')

parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--random_seed', default=6, type=int, help='This part is for controlling the random seed for reproductability')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.empty_cache()


def train_model(dataloaders, dataset_sizes, TRAIN, VAL, model, criterion, optimizer, scheduler,
                num_epochs, device, start_epoch=0, is_gist=False, is_saliency=False, is_log=False):
    args = parser.parse_args()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_R = 0.0
    best_loss = 0

    logger = Logger('./logs/' + args.logfolder)
    for epoch in range(start_epoch, num_epochs):
        logs = {}
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            name_list = []
            labels_list = []
            preds_list = []
            running_loss = 0.0

            # Iterate over data.
            for name, inputs, labels in dataloaders[phase]:
                if not is_gist and not is_saliency:
                    inputs = inputs.to(device)
                elif is_gist and is_saliency:
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)
                    inputs[2] = inputs[2].to(device)
                elif is_gist or is_saliency:
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)

                    outputs = 1 + outputs * (9 - 1)  # normalize to 1-9   Because our network last layer is sigmoid, which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
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
                    name_list.extend(name)

                    # clean the cache
                    # del inputs, labels, outputs, loss
                    # torch.cuda.empty_cache()

            if phase == TRAIN:
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_R, _ = pearsonr(labels_list, preds_list)
            epoch_R2 = np.around(epoch_R ** 2, 2)

            print('{} Loss: {:.4f} R: {:.4f} R2: {:.4f}'.format(
                phase, epoch_loss, epoch_R, epoch_R2))

            prefix = phase + '_'

            logs[prefix + 'log loss'] = epoch_loss
            logs[prefix + 'correlation'] = epoch_R

            if phase == VAL:
                best_R = epoch_R
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

                save_checkpoint(best_R, best_loss, best_epoch, model, best_model_wts,
                                optimizer,
                                os.path.join(args.model_dir, args.model_name[:-4] + '_epoch' + str(best_epoch)
                                             + '.pth'))
                print('Best epoch: {:} Best loss: {:.4f} Best R: {:.4f} '.format(
                    best_epoch, best_loss, best_R))

            '''
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
            '''

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            if is_log:
                # 1. Log scalar values (scalar summary)

                info = {prefix + 'loss': logs[prefix + 'log loss'],
                        prefix + 'correlation': logs[prefix + 'correlation']}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    if value.requires_grad:
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                        if value.grad is not None:
                            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

                # 3. Log training images (image summary)
                if not is_gist:
                    info = {prefix + 'images': inputs.view(-1, 3, 224, 224)[:10].cpu().numpy()}

                    for tag, images in info.items():
                        logger.image_summary(tag, (images * 255).astype(np.uint8), epoch + 1)

        print()

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

    torch.manual_seed(parser.parse_args().random_seed)
    random.seed(parser.parse_args().random_seed)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists('./result'):
        os.makedirs('./result')

    start_epoch = 0
    is_gist = args.is_gist
    is_saliency = args.is_saliency

    is_log = args.is_log

    if args.resume:
        if os.path.isfile(os.path.join(args.model_dir, args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(os.path.join(args.model_dir, args.resume), map_location='cuda:0')
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

    elif args.is_fine_tune:
        print("=> loading old model '{}'".format(args.file_name))
        old_model = torch.load(os.path.join(args.model_dir, args.file_name), map_location='cuda:0')
        start_epoch = args.start_epoch
        model = old_model['model']
        model.load_state_dict(old_model['state_dict'])

    else:
        if args.model_to_run == 0:
            model = VGGReg()
        elif args.model_to_run == 1:
            model = Amygdala(args.n_layers)
        elif args.model_to_run == 2:
            model = Amy_IntermediateRoad(lowfea_VGGlayer=args.lowfea_VGGlayer, highfea_VGGlayer=args.highfea_VGGlayer)
        else:
            is_gist = True
            model = Amygdala_lowroad_gist_model6v2(lowfea_VGGlayer=args.lowfea_VGGlayer,
                                                   highfea_VGGlayer=args.highfea_VGGlayer,
                                                   is_gist=is_gist)

    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=1)

    print(model)

    model = model.to(device)

    print("Test before training")

    if len(args.gabor_dir) is 2:

        dataloaders_test, dataset_sizes_test = cond_dataloader2(args.data_dir, args.TRAIN, args.TRAIN2, args.VAL, args.TEST,
                                                            args.csv_train, args.csv_train2, args.csv_val,
                                                            args.csv_test,
                                                            args.batch_size, istrain=False, is_gist=is_gist,
                                                            is_saliency=is_saliency, gabor_dir=args.gabor_dir,
                                                            manipulation=args.manipulation)

        cond_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device, is_gist, is_saliency)

        # ---------------- Training-----------------------------#
        print('-' * 10)
        print("Preparing training data and it will take a while...")
        dataloaders_train, dataset_sizes_train = cond_dataloader2(args.data_dir, args.TRAIN, args.TRAIN2, args.VAL,
                                                              args.TEST,
                                                              args.csv_train, args.csv_train2, args.csv_val,
                                                              args.csv_test,
                                                              args.batch_size, istrain=True, is_gist=is_gist,
                                                              is_saliency=is_saliency, gabor_dir=args.gabor_dir,
                                                              manipulation=args.manipulation)

        model = train_model(dataloaders_train, dataset_sizes_train, 'IAPS_Conditioning_Finetune',
                        args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
                        args.epoch, device, start_epoch, is_gist, is_saliency, is_log)

    #    model = train_model(dataloaders_train, dataset_sizes_train, args.TRAIN,
    #                        args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
    #                        args.epoch, device, start_epoch, is_gist, is_saliency, is_log)

        # ----------------test-----------------------------#
        cond_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device, is_gist, is_saliency)

    elif len(args.gabor_dir) is 1:
        dataloaders_test, dataset_sizes_test = cond_dataloader(args.data_dir, args.TRAIN, args.VAL,
                                                                args.TEST,
                                                                args.csv_train, args.csv_val,
                                                                args.csv_test,
                                                                args.batch_size, istrain=False, is_gist=is_gist,
                                                                is_saliency=is_saliency, gabor_dir=args.gabor_dir)

        cond_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device, is_gist, is_saliency)

        # ---------------- Training-----------------------------#
        print('-' * 10)
        print("Preparing training data and it will take a while...")
        dataloaders_train, dataset_sizes_train = cond_dataloader(args.data_dir, args.TRAIN, args.VAL,
                                                                  args.TEST,
                                                                  args.csv_train, args.csv_val,
                                                                  args.csv_test,
                                                                  args.batch_size, istrain=True, is_gist=is_gist,
                                                                  is_saliency=is_saliency, gabor_dir=args.gabor_dir)

        model = train_model(dataloaders_train, dataset_sizes_train, args.TRAIN,
                            args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
                            args.epoch, device, start_epoch, is_gist, is_saliency, is_log)

        #    model = train_model(dataloaders_train, dataset_sizes_train, args.TRAIN,
        #                        args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
        #                        args.epoch, device, start_epoch, is_gist, is_saliency, is_log)

        # ----------------test-----------------------------#
        cond_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device, is_gist, is_saliency)

    elif len(args.gabor_dir) > 2:
        dataloaders_test, dataset_sizes_test = cond_dataloader3(args.data_dir, args.TRAIN, args.TRAIN2, args.VAL,
                                                              args.TEST,
                                                              args.csv_train, args.csv_train2, args.csv_val,
                                                              args.csv_test,
                                                              args.batch_size, istrain=False, is_gist=is_gist,
                                                              is_saliency=is_saliency, gabor_dir1=['./data/gabor_RSA2/gabor-gaussian-45-freq5-cont100.png'],
                                                              gabor_dir2= args.gabor_dir,
                                                              manipulation=args.manipulation)

        cond_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device, is_gist, is_saliency)

        # ---------------- Training-----------------------------#
        print('-' * 10)
        print("Preparing training data and it will take a while...")
        dataloaders_train, dataset_sizes_train = cond_dataloader3(args.data_dir, args.TRAIN, args.TRAIN2, args.VAL,
                                                              args.TEST,
                                                              args.csv_train, args.csv_train2, args.csv_val,
                                                              args.csv_test,
                                                              args.batch_size, istrain=True, is_gist=is_gist,
                                                              is_saliency=is_saliency, gabor_dir1=['./data/gabor_RSA2/gabor-gaussian-45-freq5-cont100.png'],
                                                              gabor_dir2= args.gabor_dir,
                                                              manipulation=args.manipulation)

        model = train_model(dataloaders_train, dataset_sizes_train, 'IAPS_Conditioning_Finetune',
                            args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
                            args.epoch, device, start_epoch, is_gist, is_saliency, is_log)

        #    model = train_model(dataloaders_train, dataset_sizes_train, args.TRAIN,
        #                        args.VAL, model, criterion, optimizer_ft, exp_lr_scheduler,
        #                        args.epoch, device, start_epoch, is_gist, is_saliency, is_log)

        # ----------------test-----------------------------#
        cond_eval_model(dataloaders_test, dataset_sizes_test, args.TEST, model, criterion, device, is_gist, is_saliency)


if __name__ == '__main__':
    main()
