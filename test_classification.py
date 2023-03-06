from __future__ import print_function, division
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import cls_eval_model,load_checkpoint,\
    plot_tuning_performance_changes_pygal,\
    plot_tuning_performance_changes_altair,plot_tuning_performance_changes_sns
from dataloader import cls_dataloader
from models.VGG_classification import VGG_classification_tuning,VGG_classification
import argparse
from utils import find_selective_kernels_tuningvalues, define_strenth_array, get_overlap_kernels,convert_layerID_to_convName,find_selective_kernels_shuffle,get_nonoverlap_kernels
import numpy as np
import itertools
import glob
import copy
from datetime import datetime
import csv
import pandas as pd
plt.ion()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Params
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--data_dir', default='/data/seowung/data/', type=str, help='the data root folder')
parser.add_argument('--TRAIN', default='IAPS_recategory', type=str, help='the folder of training data used for training the model;')
parser.add_argument('--VAL', default='IAPS_recategory_pl_val_525', type=str, help='the folder of validation data-(optional)')
parser.add_argument('--TEST', default='IAPS_recategory_pl_test_525', type=str,
                    help='target: the folder of data for tuning or testing')

parser.add_argument('--batch_size', default=32, type=int, help='batch size')

parser.add_argument('--model_dir', default='./savedmodel', type=str, help='where to save the trained model')
parser.add_argument('--model_name',
                    default='best_model_emotion_vggtuning_lr1e2_bs128_ep50_2class_IAPS_pl_Feb242022_epoch7.pth',
                    type=str, help='name of the trained model')
#tuning
parser.add_argument('--istuning_apply', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to apply tuning')

parser.add_argument('--category', default='pleasant', type=str,
                    help='which categories to tune.  Format: A;  for example: unpleasant or pleasant'
                         'This category means the category of the kernels selective (perefering) for ')

parser.add_argument('--tuning_layer',default='1,3,6,8,11,13,15,18,20,22,25,27,29',
                    type=str, help='which layer to tune')


parser.add_argument('--tuning_kernel_ids', default='shuffle', type=str,
                    help='which kernels to tune; Format: 1.either 0,10,20,33  or  2.all， or 3.overlap, or 4.shuffle;'
                         'or 5. nonselective 6. nonoverlap;  '
                         ' when setting to all,'
                         'all kernels that are all **selective** for the category will be tuned.'
                         'when setting to overlap, the kernels are the ones having overlap across datasets;'
                         'when setting to shuffle, the target kernels are randomly selected from each layer with a fixed number of kernels.'
                         'nonselective is to choose the type of channels not in the selective pool defined by tuning category.'
                         'nonoverlap is to choose the channels not in the overlapped channels defined across two or multiple datasets'
                         ' Note: shuffle should be named as random sampling')

parser.add_argument('--tuning_kernel_shuffle_ratio', default=0.5, type=float,
                    help='the number of randomly selected tuning kernels / the total number of kernels in each layer')

parser.add_argument('--tuning_kernel_shuffle_sample_pool', default=4, type=int,
                    help='the sample pool: where to randomly select the neurons:  '
                         '1.  from all neurons in a layer;  '
                         '2. from non-selective neurons in a layer and the selective neurons are defined by one single dataset (IAPS or NAPS)'
                         '3. from non-overlapped neurons in a layer'
                        '4. from non-selective neurons in a layer and the selective neurons are the overlapped ones defined by mutiple datasets (NAPS and IAPS)')

parser.add_argument('--per_layer_overlap_csv_folder', default='./result/selectivity_overlap_IAPS_recategory_naps_recategory_Mar15_v2/', type=str,
                    help='the folder containing selectivity overlap csv files')
parser.add_argument('--per_layer_overlap_csv', default='tuningvalue_layer@_selectivity_overlap.csv',
                    type=str, help='the csv file contains whether a kernel has selectivity of overlap across datasets')

parser.add_argument('--csv_saved_folder', default='./result/IAPS_recategory/',
                    help='where to find the csv files including tuning values for all categories of each channel')

parser.add_argument('--csv_file', default='tuningvalue_layer*_IAPS_recategory.csv',
                    help='the output of csv; * represents any layer csv; this csv is used to define the tuning values not the target data of tuning;'
                         'Note: if the test dataset is not the one used for training the model, @ needs to be the name of the training dataset;'
                         'For example: if it is to use the model trained on IAPS for tuning test on naps, @ should be IAPS2 rather than naps.')

parser.add_argument('--selectivity_csv_saved_folder', default='./result/IAPS_recategory/selectivity/',
                    help='where to find the csv files including tuning value only for selective category of each channel')

parser.add_argument('--selectivity_csv_file', default='tuningvalue_layer*_@_selectivity.csv',
                    help='the output of csv; * represents any layer csv; this csv is used to define the tuning values not the target data of tuning;'
                         'Note: if the test dataset is not the one used for training the model, @ needs to be the name of the training dataset;'
                         'For example: if it is to use the model trained on IAPS2 for tuning test on naps, @ should be IAPS rather than naps.')


parser.add_argument('--tuning_strenth', default='0,5,0.1', type=str,
                    help='how much scale applied to tuning values;'
                         ' Format: start value, end value, step size; it could be like 0, 1, 0.01 '
                         'Moreover it could be like "-1, 1, 0.01" for including negative numbers;'
                         'When is_lesion is True, please use 0,1,1 (the current code is not supporting 0,0,0 for lesion')


parser.add_argument('--tuning_model', default=1, type=int,
                    help='there are three models that can guide us how to apply tuning into the selective channels '
                         'for different inputs(e.g., preferred or non_preferred)'
                         '1: input_independent. No matter the input is preferred or non-preferred, we apply (1+B*f)>1 '
                         'where B is the tuning strength and f is the tuning value; in current model, it is'
                         ' the tuning value corresponding to the tuning category;'
                         'NOTE:'  
                              'two cases:'
                                 '1. bi-directional: if setting is_minortuning_others==True and ,is_minortuning_strength_ratio ==1'
                         'all channels are in the tuning processing, it is the input-independent bi-directional case, which '
                                 'is the one mentioned in the Lindsays paper. '
                                  '2. if setting is_minortuning_others==False and ,is_minortuning_strength_ratio ==1'
                         'Positive-only: if only the attended channels are in the tuning processing, it is the case used in our emotion selectivity'
                                 'paper and in Lindays paper: positive only'

                         
                         '2: input-dependent positive-only. The negative f is used only when the input feature is not preferred. '
                         'In particular, the positive-only means we apply (1+Bf)>1 for the preferred input features'
                         ' while apply (1+Bf)=0 for the non-preferred input features to the tuning target channels. '
                         
                         '3: input-dependent bi-directional, which is to apply the tuning values that is as-is (remain original)'
                         'More particularly, when the input features are preferred by the selective '
                         'channels ( or tuning target channels), we apply (1+Bf)>1 to the target tuning channels while '
                         'when the input features are non-preferred by the selective channels,  '
                         'we apply (1+Bf)<1 or (1+Bf)>1, where f is negative or positive but samller than the largest '
                         'tuning value, to the target tuning channels. '
                        '4: randonly select one response across category to individual image; This model is used to produce shuffle response results')


parser.add_argument('--is_minortuning_others', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether tuning the other channels during tuning the channels specified by --category')

parser.add_argument('--is_minortuning_strength_ratio', default=1, type=float,
                    help='if we want to make the tuning strength for other channels smaller than the ones for the '
                         'attended channels, we need to set the value smaller than 1; '
                         'the minor ratio of the tuning strength; '
                         'the minor tuning strength is equal to this ratio * the tuning strength of the specified channels')

parser.add_argument('--is_lesion', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether remove the channels (set the feature map as zeros)')


parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--plot_extra_title', default='',
                    help='figure title extra information. Format: None or string')

parser.add_argument('--result_saved_folder', default='./result/tuning/IAPS_recategory_pl_train_525/tuning_output/nonselective_binary_tuning_pleasant_alllayers_March_2_2022_F1_stength2/',
                    help='where to save result')

parser.add_argument('--output_csv', default='tuning_test_result.csv',
                    help='Record [category, layer, performance, performance_change, tuning_category]')

parser.add_argument('--model_metric',default='F1-score', type=str, help='we have two ways to record the model performance: 1. F1-score (work for binary and multile classes)'
                                                                        ' 2. AUC (only avaiable for binary) 3. accuracy')


parser.add_argument('--is_record_tuning_detail', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to record tuning details into csv.Note: this will slow the program significantly')

def find_path_per_layer_overlap_csv(per_layer_overlap_csv_folder, per_layer_overlap_csv_ori,layer):
    if '@' in per_layer_overlap_csv_ori:
        per_layer_overlap_csv = per_layer_overlap_csv_ori.replace('@', layer)
    else:
        per_layer_overlap_csv = per_layer_overlap_csv_ori
    path_per_layer_overlap_csv = os.path.join(per_layer_overlap_csv_folder, per_layer_overlap_csv)

    path_per_layer_overlap_csv = path_per_layer_overlap_csv[:-4] + '.*csv'
    path_per_layer_overlap_csv = list(glob.iglob(path_per_layer_overlap_csv, recursive=True))[0]
    return path_per_layer_overlap_csv



if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders, dataset_sizes, class_names, class_to_idx = cls_dataloader(args.data_dir, args.TRAIN,
                                                             args.VAL, args.TEST, args.batch_size, False)

    if args.istuning_apply:
        print('Apply tuning is {} and apply lesion is {}'.format(args.istuning_apply, args.is_lesion))
        tuning_layers = args.tuning_layer.split(',')
        strenths = []
        model_performance_scores = {} #could be all layers

        now = datetime.now()
        timestamp = datetime.timestamp(now)

        if not os.path.exists(args.result_saved_folder):
            os.makedirs(args.result_saved_folder)

        output_csv = os.path.join(args.result_saved_folder, args.output_csv[:-4] + '_' + str(now) + '.csv')

        schema = ['tuning_category', 'response_category', 'layer', 'layerName',
                  'original_performance', 'high_performance', 'low_performance',
                  'performance_change']
        per_layer_overlap_csv_ori = args.per_layer_overlap_csv
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Gives the header name row into csv
            writer.writerow([g for g in schema])
            performance_changes_layers = {}

            if '@' in args.csv_file:
                csv_file_ori = args.csv_file.replace('@', args.TRAIN)
            else:
                csv_file_ori = args.csv_file
            csv_file_ori = os.path.join(args.csv_saved_folder, csv_file_ori)

            if '@' in args.selectivity_csv_file:
                selectivity_csv_file_ori = args.selectivity_csv_file.replace('@', args.TRAIN)
            else:
                selectivity_csv_file_ori = args.selectivity_csv_file
            selectivity_csv_file_ori = os.path.join(args.selectivity_csv_saved_folder, selectivity_csv_file_ori)

            kernel_category = None
            for layer in tuning_layers:
                print('\nStart tuning layer:{}'.format(layer))
                print('-' * 10)
                if '*' in csv_file_ori:
                    csv_file = csv_file_ori.replace('*', layer)
                else:
                    csv_file = csv_file_ori

                #csv_file = csv_file[:-4] + '*.csv'

                print(csv_file)
                #csv_file = list(glob.iglob(csv_file, recursive=True))[0]

                if '*' in selectivity_csv_file_ori:
                    selectivity_csv_file = selectivity_csv_file_ori.replace('*', layer)
                else:
                    selectivity_csv_file = selectivity_csv_file_ori

                selectivity_csv_file = selectivity_csv_file[:-4] + '*.csv'


                #get all of the selective channels and other non-selective channels
                tuning_kernel_ids_all, tuningvalues_all,\
                non_selective_kernels, non_selective_tuningvalues = \
                    find_selective_kernels_tuningvalues(args.category, int(layer), csv_file)
                tuning_kernel_ids = args.tuning_kernel_ids
                if args.tuning_kernel_ids == 'all': #all means all selective neurons not all neurons
                    tuning_kernel_ids_current = tuning_kernel_ids_all

                elif args.tuning_kernel_ids == 'overlap':
                    path_per_layer_overlap_csv = find_path_per_layer_overlap_csv(args.per_layer_overlap_csv_folder,
                                                                                 per_layer_overlap_csv_ori, layer)

                    tuning_kernel_ids_current = get_overlap_kernels(args.category, path_per_layer_overlap_csv)

                elif args.tuning_kernel_ids == 'shuffle':
                    selectivity_csv_file = list(glob.iglob(selectivity_csv_file, recursive=True))[0]
                    path_per_layer_overlap_csv = None
                    if args.tuning_kernel_shuffle_sample_pool in [3,4]:
                        path_per_layer_overlap_csv = find_path_per_layer_overlap_csv(args.per_layer_overlap_csv_folder, per_layer_overlap_csv_ori, layer)
                    tuning_kernel_ids_all, tuningvalues_all, \
                    non_selective_kernels, non_selective_tuningvalues, kernel_category = \
                        find_selective_kernels_shuffle(int(layer), csv_file, selectivity_csv_file=selectivity_csv_file, tuning_category=args.category,
                                                       random_ratio=args.tuning_kernel_shuffle_ratio,
                                                       pool=args.tuning_kernel_shuffle_sample_pool,
                                                       path_per_layer_overlap_csv=path_per_layer_overlap_csv)

                    tuning_kernel_ids_current = tuning_kernel_ids_all

                elif args.tuning_kernel_ids == 'nonselective':
                    print('****nonselective tuning************')
                    tmp_tuning_kernel_ids_all = tuning_kernel_ids_all
                    tmp_tuningvalues_all =  tuningvalues_all

                    tuning_kernel_ids_all = non_selective_kernels
                    tuningvalues_all = non_selective_tuningvalues
                    non_selective_kernels = tmp_tuning_kernel_ids_all
                    non_selective_tuningvalues = tmp_tuningvalues_all

                    tuning_kernel_ids_current = tuning_kernel_ids_all

                elif args.tuning_kernel_ids == 'nonoverlap':
                    print('****nonoverlap************')
                    path_per_layer_overlap_csv = find_path_per_layer_overlap_csv(args.per_layer_overlap_csv_folder,
                                                                                 per_layer_overlap_csv_ori, layer)

                    tuning_kernel_ids_current = get_nonoverlap_kernels(args.category, path_per_layer_overlap_csv)

                else:
                    tuning_kernel_ids_current = tuning_kernel_ids.split(',')

                tuningvalues = {}
                tuning_kernel_ids = []
                if args.tuning_kernel_ids == 'all' or args.tuning_kernel_ids == 'overlap':
                    tuning_kernel_ids = tuning_kernel_ids_current
                    tuningvalues = tuningvalues_all
                else:
                    for k in tuning_kernel_ids_current:
                        tuningvalues[k] = tuningvalues_all[k]
                        tuning_kernel_ids.append(k)


                if ',' in args.tuning_strenth:
                    tuning_strenth_define = args.tuning_strenth.split(',')
                else:
                    tuning_strenth_define = args.tuning_strenth.split(' ')

                strenths = define_strenth_array(tuning_strenth_define)
                model = VGG_classification_tuning(len(class_names), tuning_layer=int(layer),
                                                  tuning_kernel_ids=tuning_kernel_ids, tuningvalues=tuningvalues,
                                                  tuning_strenths=strenths, istuning_apply=args.istuning_apply,
                                                  is_lesion=args.is_lesion,
                                                  is_minortuning_others=args.is_minortuning_others,
                                                  is_minortuning_strength_ratio=args.is_minortuning_strength_ratio,
                                                  non_selective_kernels=non_selective_kernels,
                                                  non_selective_tuningvalues=non_selective_tuningvalues,
                                                  tuning_model=args.tuning_model,
                                                  tuning_category=args.category,
                                                  kernel_category=kernel_category
                                                  )
                # Load the  model
                model = load_checkpoint(model, os.path.join(args.model_dir, args.model_name), int(layer),
                                        args.istuning_apply)

                model = model.to(device)

                criterion = nn.CrossEntropyLoss()

                # ----------------test-----------------------------#
                category_idx = {}
                for cat in class_names:
                    category_idx[cat] = dataloaders[args.TEST].dataset.class_to_idx[cat]

                performance_score = cls_eval_model(dataloaders, dataset_sizes, class_names, args.TEST, model, criterion, device,
                               istunning=args.istuning_apply, tuningstrenths=strenths,
                               allcategory=class_names, allcategory_idx=category_idx,
                               tuning_category=args.category, plot_extra_title=args.plot_extra_title,
                               tuning_layer=layer, result_folder=args.result_saved_folder, model_metric=args.model_metric, is_record_tuning_detail=args.is_record_tuning_detail)
                model_performance_scores[int(layer)] = performance_score
                res = list(performance_score.keys()).index(0) #the key ==0 position
                lists = sorted(performance_score.items())  # sorted by key, return a list of tuples
                strenths, scores = zip(*lists)  # unpack a list of pairs into two tuples
                per_changes_cat = {}
                for cat in class_names:
                    cat_idx = category_idx[cat]
                    if args.model_metric == 'accuracy':
                        scores_to_plot = [s for s in scores]
                    elif args.model_metric == 'F1-score':
                        scores_to_plot = [s[cat_idx] for s in scores]
                    else:  # AUC only work for binary classification currently
                        scores_to_plot = [s for s in scores]


                    original_performance = scores_to_plot[res]

                    cp_scores = copy.deepcopy(scores_to_plot)
                    cp_scores.pop(res)# remove the  original performance value
                    high_score = max(cp_scores)
                    low_score = min(cp_scores)
                    low_score_idx = np.argmin(cp_scores)
                    if len(cp_scores[low_score_idx + 1:]) >= 1:
                        second_part_high_score = max(cp_scores[low_score_idx + 1:])
                    else:
                        second_part_high_score = 0
                    if high_score > original_performance:
                        changes = round((high_score - original_performance) * 100 / original_performance)
                        per_changes_cat[cat] = changes
                    else:
                        if second_part_high_score > low_score:
                            high_score = second_part_high_score
                            changes = round((second_part_high_score - original_performance) * 100 / original_performance)
                            per_changes_cat[cat] = changes
                        else:
                            changes = round((low_score - original_performance) * 100 / original_performance)
                            per_changes_cat[cat] = changes

                    writer.writerow([args.category, cat, layer, convert_layerID_to_convName(layer), original_performance, high_score, low_score, changes])
                performance_changes_layers[int(layer)] = per_changes_cat
                del model
                torch.cuda.empty_cache()
                print('\nCompleted tuning layer {}'.format(layer))
                print('-' * 10)

        plot_tuning_performance_changes_pygal(performance_changes_layers, class_names, args.category, args.TEST,
                                              args.result_saved_folder, plot_extra_title=args.plot_extra_title)

        data_to_plot = pd.read_csv(output_csv, header=0)
        plot_tuning_performance_changes_altair(data_to_plot, args.category, args.TEST,
                                               args.result_saved_folder, plot_extra_title=args.plot_extra_title)

        plot_tuning_performance_changes_sns(data_to_plot, args.category, args.TEST,
                                               args.result_saved_folder, args.is_lesion,
                                            plot_extra_title=args.plot_extra_title)

    else:
        model = VGG_classification(len(class_names))
        # Load the  model
        model = load_checkpoint(model, os.path.join(args.model_dir, args.model_name))

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        # ----------------test-----------------------------#
        cls_eval_model(dataloaders, dataset_sizes, class_names, args.TEST, model, criterion, device, allcategory_idx=class_to_idx, model_metric=args.model_metric)
        # print(model)


