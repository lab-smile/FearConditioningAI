
"""
This program is used for neural selectivity.  train_amygdala_classification is used for Amygdala modeling.
Author: Peng Liu (liupengufl@gmail.com)
Date:  July 23, 2020
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import cls_eval_model,save_checkpoint
from dataloader import cls_dataloader
from models.VGG_classification import VGG_classification, Amy_IntermediateRoad
import argparse
from torchsummary import summary
from logger import Logger

# plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='./data/gabor_RSA3_train', type=str, help='the data root folder')
parser.add_argument('--TRAIN', default='train', type=str, help='the folder of training data')
parser.add_argument('--VAL', default='val', type=str, help='the folder of validation data')
parser.add_argument('--TEST', default='test', type=str, help='the folder of test data')


parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--lrstepsize', default=100, type=int)

parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where to save the trained model')
parser.add_argument('--model_name', default='best_model.pth', type=str, help='name of the trained model')
parser.add_argument('--resume', default=None, type=str, help='the path to checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--islog',default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='if record log')

parser.add_argument('--logfolder', default='best_model_emotion_vggtuning_lr1e2_bs128_ep100_2class_naps_un_Fold4_March222022', type=str , help='name of the log')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids. only one gpu can be used')


def train_model(dataloaders, dataset_sizes, TRAIN, VAL, model, criterion,
                optimizer, scheduler, num_epochs, device, start_epoch=0):
    # liveloss = PlotLosses()
    args = parser.parse_args()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0
    best_epoch = 1
    if args.islog:
        logger = Logger('./logs/'+ args.logfolder)
    for epoch in range(start_epoch, num_epochs):
        logs = {}
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == TRAIN:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            prefix = ''
            prefix = phase + '_'

            logs[prefix + 'log loss'] = epoch_loss
            logs[prefix + 'accuracy'] = epoch_acc

            if phase == VAL:
                if epoch == 0 or epoch ==start_epoch:
                    best_loss = epoch_loss


            # deep copy the model
            if phase == VAL and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

                save_checkpoint(best_acc, best_loss, best_epoch, model, best_model_wts,
                                optimizer,
                                os.path.join(args.model_dir, args.model_name[:-4] + '_epoch' + str(best_epoch)
                                             + '.pth'))
                print('Best epoch: {:} Best loss: {:.4f} Best acc: {:.4f} '.format(
                    best_epoch, best_loss, best_acc))

            # liveloss.update(logs)
            # liveloss.send()

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            if args.islog:
                # 1. Log scalar values (scalar summary)

                info = {prefix + 'loss': logs[prefix + 'log loss'], prefix + 'accuracy': logs[prefix + 'accuracy']}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch +1 )

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                    if value.grad is not None:
                        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

                # 3. Log training images (image summary)
                info = {prefix + 'images': inputs.view(-1, 3, 224, 224)[:10].cpu().numpy()}

                for tag, images in info.items():
                    logger.image_summary(tag, images, epoch +1 )

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best epoch: {:}'.format(best_epoch))
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def main():
    args = parser.parse_args()
    gpu_Id = args.gpu_ids
    gpu_to_use = "cuda:"+str(gpu_Id)
    device = torch.device(gpu_to_use if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists('./result'):
        os.makedirs('./result')

    dataloaders_train, dataset_sizes_train, class_names_train, class_to_idx_train = cls_dataloader(args.data_dir, args.TRAIN, args.VAL, args.TEST,
                                                         args.batch_size, True)

    # Load the pretrained model from pytorch
    start_epoch = 0
    if args.resume:
        if os.path.isfile(os.path.join(args.model_dir, args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(os.path.join(args.model_dir, args.resume),map_location='cuda:0')
            # if "epoch" in checkpoint:
            #     start_epoch = checkpoint['epoch']
            # else:
            start_epoch = args.start_epoch
            model = checkpoint['model']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer_ft.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("=> no checkpoint (pretrained model) found at '{}'".format(args.resume))

    else:
        #model = VGG_classification(len(class_names_train))
        model = Amy_IntermediateRoad(lowfea_VGGlayer=10, highfea_VGGlayer=36)

        for param in model.parameters():
            param.requires_grad = True


    print(model)



    criterion = nn.CrossEntropyLoss()
    #optimizer_ft = optim.SGD(model.parameters(), lr=args.lr)
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.lrstepsize, gamma=0.1)

    model = model.to(device)

    # summary(model, (3, 224, 224))
    print("Test before training")
    dataloaders_test, dataset_sizes_test, class_names_test, class_to_idx_test = cls_dataloader(args.data_dir, args.TRAIN, args.VAL, args.TEST,
                                                         args.batch_size, False)
    cls_eval_model(dataloaders_test, dataset_sizes_test, class_names_test, args.TEST, model, criterion, device, allcategory_idx=class_to_idx_test)


    # ---------------- Training-----------------------------#
    trained_model = train_model(dataloaders_train, dataset_sizes_train, args.TRAIN, args.VAL, model, criterion,
                        optimizer_ft, exp_lr_scheduler, args.epoch, device, start_epoch)
    checkpoint = {'model': trained_model,
                  'state_dict': trained_model.state_dict(),
                  'optimizer': optimizer_ft.state_dict()}

    torch.save(checkpoint, os.path.join(args.model_dir, args.model_name))


    # ----------------test-----------------------------#
    cls_eval_model(dataloaders_test, dataset_sizes_test, class_names_test, args.TEST, trained_model, criterion, device, allcategory_idx=class_to_idx_test)
    # ----------------Visualize -----------------------------
    # cls_visualize_model(dataloaders_test, class_names_test, vgg16, args.TEST, 6, device)


if __name__ == '__main__':

    main()
