import imageio
import matplotlib

matplotlib.use('Agg')

import torchvision
import torch
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr
import os
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import metrics
import csv
import altair as alt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, auc
from PIL import Image, ImageDraw
from altair_saver import save
import matplotlib.pyplot as plt

import json


def imshow(inp, marker, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    fig.savefig('result/visual_result_' + '_' + now + '_' + marker + '.pdf')
    plt.pause(0.001)


def show_databatch(class_names, inputs, classes, marker):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, marker, title=[class_names[x] for x in classes])


def show_databatch_regression(inputs, classes, marker):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, marker, title=[round(x, 2) for x in classes.numpy()])


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    font = {'family': 'Calibri',
            'weight': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # SMALL_SIZE = 8
    # matplotlib.rc('font', size=SMALL_SIZE)
    # matplotlib.rc('axes', titlesize=SMALL_SIZE)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=3)
    plt.ylabel('True category')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted category \nAccuracy={:0.4f}'.format(accuracy))

    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    fig.savefig('result/conf_matrix_result_' + '_' + now + '.pdf')

    plt.show()


def plot_precision_recall(n_classes, Y_test, y_score):
    ###############################################################################
    # The average precision score in multi-label settings

    # np_labels_bn= label_binarize(np_labels, classes=[0, 1, 2])
    # np_preds_bn = label_binarize(filter_preds, classes=[0, 1, 2])

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[i], y_score[i])
        average_precision[i] = average_precision_score(Y_test[i], y_score[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    ###############################################################################
    # Plot the micro-averaged Precision-Recall curve
    # ...............................................
    #

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

    ###############################################################################
    # Plot Precision-Recall curve for each class and iso-f1 curves
    # .............................................................
    #
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()


def filter_array(np_labels, np_preds, n_classes):
    results_labels = {}
    results_preds = {}

    for i in range(n_classes):
        condition = (np_labels == i)
        results_labels[i] = label_binarize(np.extract(condition, np_labels), classes=[0, 1, 2])
        results_preds[i] = label_binarize(np.extract(condition, np_preds), classes=[0, 1, 2])

    return results_labels, results_preds


def plot_correaltion(y, y_cv, score_cv_r, score_cv_r2, mse_cv, test_folder):
    # Fit a line to the C vs response

    z = np.polyfit(y, y_cv, 1)
    sns.set()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_cv, c='red', edgecolors='k')
    # Plot the best fit line
    ax.plot(y, np.polyval(z, y), c='blue', linewidth=1)

    plt.title(test_folder + '\n $R^{2}$: %5.4f' % score_cv_r2 + ' \n  R: %5.4f' % score_cv_r + '; MSE: %5.4f' % mse_cv)

    plt.ylabel('Predicted valence rating')
    plt.xlabel('Measured valence rating')

    # set axes range
    plt.xlim(1, 9)
    plt.ylim(1, 9)

    plt.show()
    now = datetime.now()
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    fig.savefig('result/' + test_folder + '_regression_plot' + '_' + str(now) + '_' + 'R-%5.4f' % score_cv_r + '.pdf')


def plot_tuning_accuracy(strenths, accuracy, category, tuning_category, test_folder,
                         tuning_layer, result_folder, plot_extra_title=None):
    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(strenths, accuracy, color='black', linewidth=1, marker='o', markerfacecolor='black', markersize=3)

    if plot_extra_title is not None:
        title = 'Accuracy of {} when tuning {} for {} layer on dataset {} \n {} '.format(category, tuning_category,
                                                                                         tuning_layer,
                                                                                         test_folder, plot_extra_title)
    else:
        title = 'Accuracy of {} when tuning {} for {} layer on dataset {} '.format(category,
                                                                                   tuning_category, tuning_layer,
                                                                                   test_folder)

    plt.title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Tuning strenth')

    plt.show()
    now = datetime.now()
    filename = tuning_layer + 'layer_' + tuning_category + '-tuning_category_' \
               + category + '_response_' + str(now) + '.pdf'
    path_to_save_result = os.path.join(result_folder, filename)
    fig.savefig(path_to_save_result)


# plot one layer each time
def plot_tuning_accuracy_changes(performance_changes, class_names, tuning_category, test_folder, plot_extra_title=None):
    lists = sorted(performance_changes.items())  # sorted by key, return a list of tuples
    layers, per_changes = zip(*lists)  # unpack a list of pairs into two tuples

    for cat in class_names:
        scores_to_plot = [p[cat] for p in per_changes]
        sns.set()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(layers, scores_to_plot, color='black', linewidth=1, marker='o', markerfacecolor='black', markersize=3)

        if plot_extra_title is not None:
            title = 'Performance changes of {} when tuning {} on dataset {} \n {} '.format(cat, tuning_category,
                                                                                           test_folder,
                                                                                           plot_extra_title)
        else:
            title = 'Performance changes of {} when tuning {} on dataset {} '.format(cat, tuning_category,
                                                                                     test_folder
                                                                                     )

        plt.title(title)

        plt.ylabel('Performance changes (%pts)')
        plt.xlabel('Tuning layers')

        plt.xlim(min(layers), max(layers))

        plt.show()
        now = datetime.now()
        result_folder = './result/tuning/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        fig.savefig(result_folder + 'performance_changes_all_layers_' + tuning_category + '-tuning_category_'
                    + cat + '_response_' + str(now) + '.pdf')


# plot all layers
def plot_tuning_performance_changes_pygal(performance_changes, class_names, tuning_category, test_folder, result_folder,
                                          plot_extra_title=None):
    import pygal
    from pygal.style import Style

    custom_style = Style(
        colors=('#0343df', '#e50000', '#ffff14', '#929591'),
        font_family='Roboto,Helvetica,Arial,sans-serif',
        background='transparent',
        label_font_size=18,
    )
    if plot_extra_title is not None:
        title = 'Performance changes of neutral, pleasant, unpleasant when tuning {} on dataset {} \n {} '.format(
            tuning_category,
            test_folder,
            plot_extra_title)
    else:
        title = 'Performance changes of neutral, pleasant, unpleasant when tuning {} on dataset {} '.format(
            tuning_category,
            test_folder
            )
    c = pygal.Bar(
        title=title,
        style=custom_style,
        y_title='Performance changes (%pts)',
        x_title='Tuning layers',
        width=1200,
        x_label_rotation=270,
    )

    lists = sorted(performance_changes.items())  # sorted by key, return a list of tuples
    layers, per_changes = zip(*lists)  # unpack a list of pairs into two tuples
    for cat in class_names:
        scores_to_plot = [p[cat] for p in per_changes]
        c.add(cat, scores_to_plot)

    c.x_labels = layers

    # c.render_to_file('pygal.svg')
    now = datetime.now()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = 'performance_changes_all_layers_' + tuning_category + '-tuning_category_' + 'all_emotions_response_' + str(
        now) + '.svg'
    path_to_save_result = os.path.join(result_folder, filename)
    c.render_to_file(path_to_save_result)


def plot_tuning_performance_changes_altair(data_to_plot, tuning_category, test_folder, result_folder,
                                           plot_extra_title=None):
    alt.renderers.enable('altair_viewer')

    df = pd.DataFrame({'layer': data_to_plot['layerName'],
                       'per_ch': data_to_plot['performance_change'],
                       'category': data_to_plot['response_category']
                       })

    res_categories = np.unique(data_to_plot['response_category'].values)
    if res_categories.size == 3:
        cmap = {
            'neutral': '#0343df',
            'pleasant': '#e50000',
            'unpleasant': '#ffff14'
        }
        cat_names = res_categories[0], res_categories[1], \
                    res_categories[2]
    else:
        cmap = {
            res_categories[0]: assign_color(res_categories[0]),
            res_categories[1]: assign_color(res_categories[1]),
        }

        cat_names = res_categories[0], res_categories[1]

    if plot_extra_title is not None:
        title = 'Performance changes of {} when tuning {} on dataset {} \n {} '.format(cat_names, tuning_category,
                                                                                       test_folder,
                                                                                       plot_extra_title)
    else:
        title = 'Performance changes of {} when tuning {} on dataset {} '.format(cat_names, tuning_category,
                                                                                 test_folder
                                                                                 )
    df['layer'] = df['layer'].astype(str)

    # We're still assigning, e.g. 'party' to x, but now we've wrapped it
    # in alt.X in order to specify its styling
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('category', title=None),
        y=alt.Y('per_ch', title='Performance changes (% pts)'),
        column=alt.Column('layer', sort=list(df['category']), title='Layers'),
        color=alt.Color('category', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values())))
    ).properties(title=title)

    # c.render_to_file('pygal.svg')
    now = datetime.now()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    filename = 'performance_changes_all_layers_' + tuning_category + '-tuning_category_' \
               + 'all_emotions_response_' + str(now) + '.html'
    path_to_save_result = os.path.join(result_folder, filename)
    chart.save(path_to_save_result)


def plot_tuning_performance_changes_sns(data_to_plot, tuning_category, test_folder,
                                        result_folder, is_lesoin=False, plot_extra_title=None):
    sns.set(style="ticks", color_codes=True)
    sns.set(font='Calibri')
    sns.set(font_scale=1.5)

    res_categories = np.unique(data_to_plot['response_category'].values)
    if res_categories.size == 3:
        cat_names = res_categories[0], res_categories[1], \
                    res_categories[2]
    else:
        cat_names = res_categories[0], res_categories[1]

    if is_lesoin:
        title_sub = 'Lesion'
    else:
        title_sub = 'Tuning'
    if plot_extra_title is not None:
        # title = 'Performance changes of {} when tuning {} on dataset {} \n {} '.format(cat_names, tuning_category,
        #                                                                                test_folder,
        #                                                                                plot_extra_title)
        title = title_sub + ' {} \n {}'.format(tuning_category, plot_extra_title)
    else:
        title = title_sub + ' {}'.format(tuning_category)
        # title = 'Performance changes of {} when tuning {} on dataset {} '.format(cat_names, tuning_category,
        #                                                                          test_folder
        #                                                                          )

    g = sns.catplot(x="layerName", y="performance_change", hue="response_category", kind="bar",
                    legend_out=False, aspect=3, data=data_to_plot)

    g.set_axis_labels("", "Performance changes (% pts)")

    plt.title(title)
    # g.set(ylim=(-100,0))
    g._legend.set_title('')

    now = datetime.now()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    filename = 'performance_changes_all_layers_' + tuning_category + '-tuning_category_' \
               + 'all_emotions_response_' + str(now) + '_sns_.pdf'

    path_to_save_result = os.path.join(result_folder, filename)
    g.savefig(path_to_save_result)


def convert_layerID_to_convName(layer):
    switcher = {
        1: "conv1",
        3: "conv2",
        6: "conv3",
        8: "conv4",
        11: "conv5",
        13: "conv6",
        15: "conv7",
        18: "conv8",
        20: "conv9",
        22: "conv10",
        25: "conv11",
        27: "conv12",
        29: "conv13"
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(int(layer), "nothing")


def assign_color(category):
    if 'unpleasant' == category:
        color = '#ffff14'
    elif 'pleasant' == category:
        color = '#e50000'
    elif 'non_unpleasant' == category:
        color = '#AF7AC5'
    elif 'non_pleasant' == category:
        color = '#2ECC71'
    elif 'neutral' == category:
        color = '#0343df'
    else:
        color = '#566573'
    return color


def plot_selectivity_channel_overlabp_altair(data_to_plot, result_folder, overlap_with, plot_extra_title=None):
    alt.renderers.enable('altair_viewer')

    df = pd.DataFrame({'layer': data_to_plot['layer'],
                       overlap_with: data_to_plot[overlap_with],
                       'category': data_to_plot['category']
                       })

    res_categories = np.unique(data_to_plot['category'].values)
    if res_categories.size == 3:
        cmap = {
            'neutral': '#0343df',
            'pleasant': '#e50000',
            'unpleasant': '#ffff14'
        }
        cat_names = res_categories[0], res_categories[1], \
                    res_categories[2]
    elif res_categories.size == 2:
        cmap = {
            res_categories[0]: assign_color(res_categories[0]),
            res_categories[1]: assign_color(res_categories[1]),
        }
        cat_names = res_categories[0], res_categories[1]
    else:
        cmap = {
            res_categories[0]: assign_color(res_categories[0]),
        }
        cat_names = res_categories[0]

    if plot_extra_title is not None:
        title = 'Overlap with {} of selective channels for {} \n {}'.format(overlap_with, cat_names, plot_extra_title)
    else:
        title = 'Overlap with {} of selective channels for {} '.format(overlap_with, cat_names)

    df['layer'] = df['layer'].astype(str)

    # We're still assigning, e.g. 'party' to x, but now we've wrapped it
    # in alt.X in order to specify its styling
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('category', title=None),
        y=alt.Y(overlap_with, title='Overlap (% pts)'),
        column=alt.Column('layer', sort=list(df['category']), title='Layers'),
        color=alt.Color('category', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values())))
    ).properties(title=title)

    # c.render_to_file('pygal.svg')
    now = datetime.now()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    filename = overlap_with + '_selectivity_overlap_summary_' + str(now) + '.html'
    path_to_save_result = os.path.join(result_folder, filename)
    chart.save(path_to_save_result)


def plot_num_channel_within_layers(data_to_plot, result_folder, plot_extra_title=None):
    sns.set(style="ticks", color_codes=True)
    sns_plot = sns.catplot(x="layer", y="num_channels", hue="category", kind="bar", data=data_to_plot)

    now = datetime.now()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = 'number_selectivity_channels_within_layers_' + str(now) + '.pdf'
    path_to_save_result = os.path.join(result_folder, filename)
    sns_plot.savefig(path_to_save_result)


# for classification
def cls_visualize_model(dataloaders, class_names, vgg, test_folder, num_images, device):
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[test_folder]):
        inputs, labels = data
        size = inputs.size()[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]

        print("Ground truth:")
        show_databatch(class_names, inputs.data.cpu(), labels.data.cpu(), 'Ground truth')
        print("Prediction:")
        show_databatch(class_names, inputs.data.cpu(), predicted_labels, 'Prediction')

        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()

        images_so_far += size
        if images_so_far >= num_images:
            break

    vgg.train(mode=was_training)  # Revert model back to original training state


def cls_process_output(criterion, outputs, labels, loss_test, acc_test, labels_list, preds_list, i, preds_proba_list,
                       positive_class_idx=1, istuning=False, strenth=None):
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, preds = torch.max(outputs.data, 1)

    loss = criterion(outputs, labels)

    loss_test += loss.data
    acc_test += torch.sum(preds == labels.data)

    if istuning:
        print(i + 1, strenth, 'loss: {:.4f}'.format(loss.cpu().item()))
    else:
        print(i + 1, 'loss: {:.4f}'.format(loss.cpu().item()))

    labels_list.extend(labels.cpu().numpy())
    preds_list.extend(preds.cpu().numpy())
    #### keep probabilities for the positive outcome only how to find which dim is the positive class ????
    preds_proba_list.extend(probabilities[:, 1].detach().cpu().numpy())
    return loss_test, acc_test, labels_list, preds_list, preds_proba_list


def print_result(loss_test, acc_test, data_size, class_names, labels_list, preds_list, preds_proba_list):
    # calculate accuracy
    avg_loss = loss_test / data_size
    avg_acc = acc_test.double() / data_size
    # plot confusion matrix
    conf_matr = confusion_matrix(labels_list, preds_list)
    plot_confusion_matrix(conf_matr, normalize=False, target_names=class_names, title="Confusion Matrix")

    # auc_score = roc_auc_score(labels_list, preds_list)
    # acc_score = accuracy_score(labels_list, preds_list)

    fpr, tpr, thresholds = roc_curve(labels_list, preds_proba_list, pos_label=2)
    auc_score = auc(fpr, tpr)

    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print("ROC AUC (test):{:.4f}".format(auc_score))

    print(conf_matr)
    print('-' * 10)
    report = classification_report(labels_list, preds_list, digits=4)
    print(report)
    print('-' * 10)

    # # Print Area Under Curve
    plt.figure(figsize=[8, 6])
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(fpr, tpr, 'b', label='ROC curve (area = %0.4f)' % auc_score)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    now = datetime.now()
    plt.savefig('./result/roc_' + str(now) + '.pdf')
    plt.show()


# for classification
def cls_eval_model(dataloaders, dataset_sizes, class_names, test_folder, vgg, criterion, device,
                   istunning=False, tuningstrenths=None, allcategory=None,
                   allcategory_idx=None, tuning_category=None,
                   plot_extra_title=None, tuning_layer=None, result_folder=None, model_metric='AUC',
                   is_record_tuning_detail=True):
    since = time.time()
    loss_test = 0
    acc_test = 0
    labels_list = []
    preds_list = []
    preds_proba_list = []
    if istunning:
        tuning_labels_list = {}
        tuning_preds_list = {}
        tuning_loss_test = {}
        tuning_acc_test = {}
        tuning_preds_proba_list = {}

        for strenth in tuningstrenths:
            tuning_labels_list[strenth] = []
            tuning_preds_list[strenth] = []
            tuning_preds_proba_list[strenth] = []
            tuning_loss_test[strenth] = 0
            tuning_acc_test[strenth] = 0

    test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        if i % 100 == 0:
            print("\rTest batch {}/{} \n".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        positive_class_idx = 1
        for c in class_names:
            if 'non' not in c:
                positive_class_idx = allcategory_idx[c]
                break

        if not istunning:
            outputs = vgg(inputs)
            loss_test, acc_test, labels_list, preds_list, preds_proba_list = cls_process_output(
                criterion, outputs, labels, loss_test, acc_test, labels_list, preds_list, i, preds_proba_list,
                positive_class_idx=positive_class_idx)
        else:
            labels_names = [class_names[x] for x in labels]
            outputs = vgg(inputs, labels_names)
            for strenth in tuningstrenths:
                currrent_output = outputs[strenth]
                tuning_loss_test[strenth], tuning_acc_test[strenth], tuning_labels_list[strenth], \
                tuning_preds_list[strenth], tuning_preds_proba_list[strenth] = cls_process_output(
                    criterion, currrent_output, labels, tuning_loss_test[strenth],
                    tuning_acc_test[strenth], tuning_labels_list[strenth], tuning_preds_list[strenth], i,
                    tuning_preds_proba_list[strenth],
                    positive_class_idx=positive_class_idx, istuning=True, strenth=strenth)

        del inputs, labels, outputs
        torch.cuda.empty_cache()

    elapsed_time = time.time() - since
    print()

    if not istunning:
        print_result(loss_test, acc_test, dataset_sizes[test_folder], class_names, labels_list, preds_list,
                     preds_proba_list)
    else:
        performance_scores = {}
        for strenth in tuningstrenths:
            print('\n Tuning strenth:' + str(strenth))
            print_result(tuning_loss_test[strenth], tuning_acc_test[strenth],
                         dataset_sizes[test_folder], class_names, tuning_labels_list[strenth],
                         tuning_preds_list[strenth], tuning_preds_proba_list[strenth])
            if model_metric == 'AUC':  # we may need to get the prediction probability  (currently, it is binary available))
                fpr, tpr, _ = roc_curve(tuning_labels_list[strenth], tuning_preds_proba_list[strenth])
                clf_rep = auc(fpr, tpr)
                performance_scores[strenth] = clf_rep
            elif model_metric == "accuracy":
                clf_rep = accuracy_score(tuning_labels_list[strenth], tuning_preds_list[strenth])
                performance_scores[strenth] = clf_rep
            else:  # F1-score (can work multiple classes)
                clf_rep = metrics.precision_recall_fscore_support(tuning_labels_list[strenth],
                                                                  tuning_preds_list[strenth])
                performance_scores[strenth] = clf_rep[2].round(4)

        if is_record_tuning_detail:
            details_csv_schema = ['layer', 'tuning_strength', 'tuning_category', 'response_category', 'new_performance']
            if len(performance_scores) > 1:
                lists = sorted(performance_scores.items())  # sorted by key, return a list of tuples
                strenths, scores = zip(*lists)  # unpack a list of pairs into two tuples

                now = datetime.now()
                details_tuning_csv_name = tuning_layer + 'layer_' + tuning_category + '-tuning_category_' + str(
                    now) + '.csv'
                details_tuning_csv = os.path.join(result_folder, details_tuning_csv_name)

                with open(details_tuning_csv, 'w', newline='') as details_file:
                    writer_detail = csv.writer(details_file)
                    # Gives the header name row into csv
                    writer_detail.writerow([g for g in details_csv_schema])
                    for cat in allcategory:
                        cat_idx = allcategory_idx[cat]
                        if model_metric == 'accuracy':
                            scores_to_plot = [s for s in scores]
                        elif model_metric == 'F1-score':
                            scores_to_plot = [s[cat_idx] for s in scores]
                        else:  # AUC   only available for binary classification
                            scores_to_plot = [s for s in scores]

                        plot_tuning_accuracy(strenths, scores_to_plot, cat,
                                             tuning_category, test_folder, tuning_layer, result_folder,
                                             plot_extra_title)

                        for i in range(len(strenths)):
                            writer_detail.writerow(
                                [tuning_layer, strenths[i], tuning_category, cat, scores_to_plot[i]])


        else:
            for cat in allcategory:

                score = performance_scores[tuningstrenths[0]]
                cat_idx = allcategory_idx[cat]
                if model_metric == 'AUC':
                    print('Tuning {}: {} score {}'.format(tuning_category, cat, score))
                elif model_metric == "accuracy":
                    print('Tuning {}: {} score {}'.format(tuning_category, cat, score))
                else:
                    print('Tuning {}: {} score {}'.format(tuning_category, cat, score[cat_idx]))

    print("\nEvaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    if istunning and len(performance_scores) > 1:
        return performance_scores


# for regression
def reg_visualize_model(dataloaders, vgg, test_folder, num_images, device):
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[test_folder]):
        _, inputs, labels = data
        size = inputs.size()[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        # _, preds = torch.max(outputs.data, 1)
        predicted_labels = outputs.view(labels.size())  # [outputs[j] for j in range(inputs.size()[0])]

        print("Ground truth:")
        if num_images <= inputs.shape[0]:
            show_databatch_regression(inputs.data[:num_images].cpu(), labels.data[:num_images].cpu(), 'Ground truth')
        else:
            show_databatch_regression(inputs.data.cpu(), labels.data.cpu(), 'Ground truth')
        print("Prediction:")
        if num_images <= inputs.shape[0]:
            show_databatch_regression(inputs.data[:num_images].cpu(), predicted_labels[:num_images].detach().cpu(),
                                      'Prediction')
        else:
            show_databatch_regression(inputs.data.cpu(), predicted_labels, 'Prediction')

        del inputs, labels, outputs, predicted_labels
        torch.cuda.empty_cache()

        images_so_far += size
        if images_so_far >= num_images:
            break

    vgg.train(mode=was_training)  # Revert model back to original training state


# for regression
def reg_eval_model(dataloaders, dataset_sizes, test_folder, vgg, criterion, device):
    since = time.time()
    loss_test = 0
    labels_list = []
    preds_list = []

    # test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        # if i % 100 == 0:
        #     print("\rTest batch {}/{} \n".format(i, test_batches), end='', flush=True)
        # print(i, data['image'].size(), data['lable'].size())

        _, inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        vgg.train(False)
        vgg.eval()
        outputs = vgg(inputs)
        outputs = 1 + outputs * (
                    9 - 1)  # normalize to 1-9   Because our network last layer is sigmoid, which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
        loss = criterion(outputs.view(labels.size()), labels.float())

        loss_test += loss.data

        # print(i+1, '{:.4f}'.format(loss.cpu().item()))

        # here taking lots of time to fix the GPU out of Memory issue: need to  be numpy()
        labels_list.extend(labels.cpu().numpy())
        preds = outputs.reshape(labels.shape).cpu().detach().numpy()
        preds_list.extend(preds)

        # clean the cache
        del inputs, labels, outputs, loss, preds
        torch.cuda.empty_cache()

    # calculate correalation R & mean valence
    original_mean_valence = np.mean(labels_list)
    decoded_mean_valence = np.mean(preds_list)
    avg_loss = loss_test / dataset_sizes[test_folder]
    score_c_r, _ = pearsonr(labels_list, preds_list)
    score_cv_r2 = np.around(score_c_r ** 2, 2)  # r2_score(labels_list, preds_list)

    # mse_loss = mean_squared_error(labels_list, preds_list)

    elapsed_time = time.time() - since
    # print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Mean value of original valence {:.4f}".format(original_mean_valence))
    print("Mean value of model's decoded valence {:.4f}".format(decoded_mean_valence))
    print("Avg MSE loss (test): {:.4f}".format(avg_loss))
    print("R (test): {:.4f}".format(score_c_r))
    print('R2 : %5.4f' % score_cv_r2)

    # plot correatlion

    plot_correaltion(labels_list, preds_list, score_c_r, score_cv_r2, avg_loss, test_folder)

    print('-' * 10)
    return score_c_r


def cond_eval_model(dataloaders, dataset_sizes, test_folder, vgg, criterion, device, is_gist=False, is_saliency=False):
    since = time.time()
    loss_test = 0
    labels_list = []
    preds_list = []

    # test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        # if i % 100 == 0:
        #     print("\rTest batch {}/{} \n".format(i, test_batches), end='', flush=True)
        # print(i, data['image'].size(), data['lable'].size())

        _, inputs, labels = data

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

        vgg.train(False)
        vgg.eval()
        outputs = vgg(inputs)
        outputs = 1 + outputs * (
                    9 - 1)  # normalize to 1-9   Because our network last layer is sigmoid, which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
        loss = criterion(outputs.view(labels.size()), labels.float())

        loss_test += loss.data

        # print(i+1, '{:.4f}'.format(loss.cpu().item()))

        # here taking lots of time to fix the GPU out of Memory issue: need to  be numpy()
        labels_list.extend(labels.cpu().numpy())
        preds = outputs.reshape(labels.shape).cpu().detach().numpy()
        preds_list.extend(preds)

        # clean the cache
        del inputs, labels, outputs, loss, preds
        torch.cuda.empty_cache()

    # calculate correalation R & mean valence
    original_mean_valence = np.mean(labels_list)
    decoded_mean_valence = np.mean(preds_list)
    avg_loss = loss_test / dataset_sizes[test_folder]
    score_c_r, _ = pearsonr(labels_list, preds_list)
    score_cv_r2 = np.around(score_c_r ** 2, 2)  # r2_score(labels_list, preds_list)

    # mse_loss = mean_squared_error(labels_list, preds_list)

    elapsed_time = time.time() - since
    # print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Mean value of original valence {:.4f}".format(original_mean_valence))
    print("Mean value of model's decoded valence {:.4f}".format(decoded_mean_valence))
    print("Avg MSE loss (test): {:.4f}".format(avg_loss))
    print("R (test): {:.4f}".format(score_c_r))
    print('R2 : %5.4f' % score_cv_r2)
    # print('MSE CV: %5.4f' % mse_loss)

    # plot correatlion

    plot_correaltion(labels_list, preds_list, score_c_r, score_cv_r2, avg_loss, test_folder)

    print('-' * 10)
    return score_c_r

def cond_eval_model2(dataloaders, dataset_sizes, test_folder, vgg, criterion, device, is_gist=False, is_saliency=False):
    since = time.time()
    loss_test = 0
    labels_list = []
    preds_list = []

    # test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        # if i % 100 == 0:
        #     print("\rTest batch {}/{} \n".format(i, test_batches), end='', flush=True)
        # print(i, data['image'].size(), data['lable'].size())

        _, inputs, labels = data

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

        vgg.train(False)
        vgg.eval()
        outputs = vgg(inputs)
        outputs = 1 + outputs * (
                    90 - 0)  # normalize to 1-9   Because our network last layer is sigmoid, which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
        loss = criterion(outputs.view(labels.size()), labels.float())

        loss_test += loss.data

        # print(i+1, '{:.4f}'.format(loss.cpu().item()))

        # here taking lots of time to fix the GPU out of Memory issue: need to  be numpy()
        labels_list.extend(labels.cpu().numpy())
        preds = outputs.reshape(labels.shape).cpu().detach().numpy()
        preds_list.extend(preds)

        # clean the cache
        del inputs, labels, outputs, loss, preds
        torch.cuda.empty_cache()

    # calculate correalation R & mean valence
    original_mean_valence = np.mean(labels_list)
    decoded_mean_valence = np.mean(preds_list)
    avg_loss = loss_test / dataset_sizes[test_folder]
    score_c_r, _ = pearsonr(labels_list, preds_list)
    score_cv_r2 = np.around(score_c_r ** 2, 2)  # r2_score(labels_list, preds_list)

    # mse_loss = mean_squared_error(labels_list, preds_list)

    elapsed_time = time.time() - since
    # print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Mean value of original valence {:.4f}".format(original_mean_valence))
    print("Mean value of model's decoded valence {:.4f}".format(decoded_mean_valence))
    print("Avg MSE loss (test): {:.4f}".format(avg_loss))
    print("R (test): {:.4f}".format(score_c_r))
    print('R2 : %5.4f' % score_cv_r2)
    # print('MSE CV: %5.4f' % mse_loss)

    # plot correatlion

    plot_correaltion(labels_list, preds_list, score_c_r, score_cv_r2, avg_loss, test_folder)

    print('-' * 10)
    return score_c_r


def save_checkpoint(best_R, best_loss, best_epoch, model, best_state_dict, optimizer, model_name):
    checkpoint = {'model': model,
                  'epoch': best_epoch,
                  'best_per': best_R,
                  'best_loss': best_loss,
                  'state_dict': best_state_dict,
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, model_name)


# for test
def load_checkpoint(model, filepath, tuning_layer=None, istuning=False):
    checkpoint = torch.load(filepath, map_location='cuda:0')
    state_dict = checkpoint['state_dict']

    if istuning and tuning_layer is not None:

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            layer_idx = k.split('.')[-2]

            if tuning_layer < 30:
                if int(layer_idx) > tuning_layer:
                    k = k.replace('features', 'rest_conv_part_net')
            else:  # fully connected layers
                layer_idx2 = k.split('.')[-3]
                if layer_idx2.isdigit():
                    layer_idx = int(layer_idx2) if int(layer_idx2) > int(layer_idx) else layer_idx
                if 'classifier' in k and int(layer_idx) + 32 > tuning_layer:
                    k = k.replace('classifier', 'rest_fc_part_net')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
    else:
        # model = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)

    for parameter in model.parameters():
        parameter.requires_grad = False

    # if 'best_R' in checkpoint:
    #     print("Best R: {:.4f}".format(checkpoint['best_R']))
    # if 'best_per' in checkpoint:
    #     print("Best peformance: {:.4f}".format(checkpoint['best_per']))
    # if 'best_loss' in checkpoint:
    #     print("Best loss: {:.4f}".format(checkpoint['best_loss']))
    # if 'epoch' in checkpoint:
    #     print("Best epoch: {:}".format(checkpoint['epoch']))

    model.eval()
    return model


from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# this is the model one case,
# where the tuning value is returned with the largest tuning value (or the selective categorial  one)
def find_selective_kernels(category, tuning_layer, csv_file):
    categories_tuning = category.split(',')

    label_csv = pd.read_csv(csv_file, header=0)
    label_csv = label_csv[label_csv['layer'] == tuning_layer]

    all_kernels = label_csv.iloc[:]['kernel']
    all_kernels = list(dict.fromkeys(all_kernels))  # remove duplicates

    selective_kernels = []
    selective_tuningvalues = {}
    non_selective_kernels = []
    non_selective_tuningvalues = {}
    for k in all_kernels:
        subset = label_csv[label_csv['kernel'] == k]
        len_subset = len(subset)
        largest_tuningvalue = 0
        largest_category = ''
        for i in range(len_subset):
            tuningvalue = float(subset.iloc[i][5])
            if tuningvalue > largest_tuningvalue:
                largest_tuningvalue = tuningvalue
                largest_category = subset.iloc[i][0]

        if largest_category in categories_tuning:  # only save the selective channels
            selective_kernels.append(k)
            selective_tuningvalues[k] = largest_tuningvalue
        else:
            non_selective_kernels.append(k)
            non_selective_tuningvalues[k] = largest_tuningvalue

    return selective_kernels, selective_tuningvalues, non_selective_kernels, non_selective_tuningvalues


# This function returns the tuning values including both preferred and non-preferred tuning values
#  one channel may have multiple tuning values, which is the difference from previous function;
def find_selective_kernels_tuningvalues(category, tuning_layer, csv_file):
    categories_tuning = category.split(',')

    label_csv = pd.read_csv(csv_file, header=0)
    label_csv = label_csv[label_csv['layer'] == tuning_layer]

    all_kernels = label_csv.iloc[:]['kernel']
    all_kernels = list(dict.fromkeys(all_kernels))  # remove duplicates

    selective_kernels = []
    selective_tuningvalues = {}
    non_selective_kernels = []
    non_selective_tuningvalues = {}
    for k in all_kernels:
        subset = label_csv[label_csv['kernel'] == k]
        len_subset = len(subset)
        largest_tuningvalue = 0
        largest_category = ''
        tuningvalue_dic = {}  # one channel may have multiple tuning values
        for i in range(len_subset):
            tuningvalue = float(subset.iloc[i][5])
            category = subset.iloc[i][0]
            tuningvalue_dic[category] = tuningvalue
            if tuningvalue > largest_tuningvalue:
                largest_tuningvalue = tuningvalue
                largest_category = category

        if largest_category in categories_tuning:  # only save the selective channels
            selective_kernels.append(k)
            selective_tuningvalues[k] = tuningvalue_dic
        else:
            non_selective_kernels.append(k)
            non_selective_tuningvalues[k] = tuningvalue_dic

    return selective_kernels, selective_tuningvalues, non_selective_kernels, non_selective_tuningvalues


# randomly select a fixed number of kernels for tuning;
def find_selective_kernels_shuffle(tuning_layer, csv_file, selectivity_csv_file=None,
                                   tuning_category=None, random_ratio=None, pool=1, path_per_layer_overlap_csv=None):
    label_csv = pd.read_csv(csv_file, header=0)
    label_csv = label_csv[label_csv['layer'] == tuning_layer]

    all_kernels = label_csv.iloc[:]['kernel']
    all_kernels = list(dict.fromkeys(all_kernels))  # remove duplicates

    if selectivity_csv_file is not None and tuning_category is not None:
        label_csv_selective = pd.read_csv(selectivity_csv_file, header=0)

    if pool == 1:  # random sample pool is all neurons in a layer
        samplepool = label_csv.iloc[:]['kernel']
    elif pool in [2, 4] and tuning_category is not None:  # random sample pool is the non-selective neurons
        label_csv_non_selective_channels = label_csv_selective[label_csv_selective['category'] != tuning_category]
        samplepool = label_csv_non_selective_channels.iloc[:]['kernel']
    elif pool == 3 and tuning_category is not None:  # random sample pool is the non-overlapped neurons
        if path_per_layer_overlap_csv is not None:
            overlap_kernels = get_overlap_kernels(tuning_category, path_per_layer_overlap_csv)
            samplepool = [i for i in all_kernels if i not in overlap_kernels]
        else:
            print('path_per_layer_overlap_csv is None!!!')
    else:
        samplepool = label_csv.iloc[:]['kernel']

    samplepool = list(dict.fromkeys(samplepool))  # remove duplicates

    selective_kernels = []
    selective_tuningvalues = {}
    non_selective_kernels = []
    non_selective_tuningvalues = {}
    kernel_category = {}

    total_num_kernels = len(all_kernels)
    if pool != 4:
        if selectivity_csv_file is not None and tuning_category is not None:
            label_csv_selective_channels = label_csv_selective[label_csv_selective['category'] == tuning_category]
            select_num_kernels = len(label_csv_selective_channels)
            print("\nThe number of selective channels to tune: %d" % select_num_kernels)
        else:
            select_num_kernels = int(round(total_num_kernels * random_ratio))
    else:
        if path_per_layer_overlap_csv is not None:
            overlap_kernels = get_overlap_kernels(tuning_category, path_per_layer_overlap_csv)
            select_num_kernels = len(overlap_kernels)
            print("\nThe number of overlapped channels: %d" % select_num_kernels)
        else:
            print('path_per_layer_overlap_csv is None!!!')

    if select_num_kernels <= len(samplepool):
        selected_kernels = random.sample(samplepool, k=select_num_kernels)
    else:
        selected_kernels = samplepool  # random.sample(samplepool, k= len(samplepool))

    for k in all_kernels:
        subset = label_csv[label_csv['kernel'] == k]
        len_subset = len(subset)

        tuningvalue_dic = {}  # one channel may have multiple tuning values
        for i in range(len_subset):
            tuningvalue = float(subset.iloc[i][5])
            category = subset.iloc[i][0]
            tuningvalue_dic[category] = tuningvalue

        if k in selected_kernels:
            selective_kernels.append(k)
            selective_tuningvalues[k] = tuningvalue_dic
        else:
            non_selective_kernels.append(k)
            non_selective_tuningvalues[k] = tuningvalue_dic

        kernel_category[k] = label_csv_selective[label_csv_selective['kernel'] == k]['category']

    return selective_kernels, selective_tuningvalues, non_selective_kernels, non_selective_tuningvalues, kernel_category


def kernel_selectivity(tuning_layer, csv_file, kernel_id):
    label_csv = pd.read_csv(csv_file, header=0)
    label_csv = label_csv[label_csv['layer'] == tuning_layer]

    subset = label_csv[label_csv['kernel'] == kernel_id]
    len_subset = len(subset)
    largest_tuningvalue = 0
    largest_category = ''
    for i in range(len_subset):
        tuningvalue = float(subset.iloc[i][5])
        if tuningvalue > largest_tuningvalue:
            largest_tuningvalue = tuningvalue
            largest_category = subset.iloc[i][0]

    return largest_category, largest_tuningvalue


# read tuning value from files
def get_tunningValue(self, category, layer, kernel, csv_file):
    label_csv = pd.read_csv(csv_file, header=0, dtype=str)
    len_csv = len(label_csv)
    tuningvalue = 0
    for i in range(len_csv):
        if label_csv.iloc[i][0] == category and int(label_csv.iloc[i][1]) == int(layer) and int(
                label_csv.iloc[i][2]) == int(kernel):
            tuningvalue = float(label_csv.iloc[i][5])
            break

    return tuningvalue


def define_strenth_array(tuning_strenth_define):
    start_value = float(tuning_strenth_define[0])
    end_value = float(tuning_strenth_define[1])
    step_size = float(tuning_strenth_define[2])
    lenth_round = len(str(step_size).split('.')[1])
    if end_value > start_value:
        all_strenths = np.round(np.arange(start_value, end_value + step_size, step_size), lenth_round)
    else:
        all_strenths = [start_value]
    return all_strenths


def get_overlap_kernels(category, overlap_csv):
    df = pd.read_csv(overlap_csv, header=0)
    overlap_df = df[df['is_overlap'] == 1]
    overlap_df_cat = overlap_df[overlap_df['category'] == category]
    kernels = overlap_df_cat['kernel'].tolist()
    kernels = list(dict.fromkeys(kernels))
    return kernels


def get_nonoverlap_kernels(category, overlap_csv):
    df = pd.read_csv(overlap_csv, header=0)
    nonoverlap_df = df[df['is_overlap'] == 0]
    overlap_df_cat = nonoverlap_df[nonoverlap_df['category'] == category]
    kernels = overlap_df_cat['kernel'].tolist()
    kernels = list(dict.fromkeys(kernels))
    return kernels


class Quadrant_Processing(object):
    r"""This part of the code is used in transforms.Compose function in Pytorch.

       Args:
            train_folder: the directory of training images of the dataset
            val_folder: the directory of validation images of the dataset
            test_folder: the directory of validation images of the dataset
            is_gist: if you set to True, it will apply gist transformation
            is_saliency: if you set to True it will apply saliency transformation

       Example:
           data_transforms = data_transform(train_folder, val_folder, test_folder)
       """

    def __init__(self, location):
        self.location = location

    def __call__(self, x):
        image = x
        height, width = x.size

        background = Image.new('RGB', (height, width))

        new_h, new_w = int(height / 2), int(width / 2)
        img = image.resize((new_h, new_w))

        if self.location == 1:
            background.paste(img, (new_h, 0))

        elif self.location == 2:
            background.paste(img)

        elif self.location == 3:
            background.paste(img, (0, new_w))

        elif self.location == 4:
            background.paste(img, (new_h, new_w))

        return background


class Quadrant_Processing_Conditioning(object):
    def __init__(self, location, patch):
        self.location = location
        self.patch = patch
        #self.patch = Image.open(random.choice(patch))

    def __call__(self, x):
        image = x
        height, width = x.size

        patch = Image.open(random.choice(self.patch))

        background = Image.new('RGB', (height, width))
        background.paste(patch)

        new_h, new_w = int(height / 2), int(width / 2)
        img = image.resize((new_h, new_w))

        if self.location == 1:
            background.paste(img, (new_h, 0))

        elif self.location == 2:
            background.paste(img)

        elif self.location == 3:
            background.paste(img, (0, new_w))

        elif self.location == 4:
            background.paste(img, (new_h, new_w))

        return background


class Quadrant_Processing_Conditioning_block_US(object):
    def __init__(self, apply):
        self.apply = apply

    def __call__(self, x):
        original_image = x
        if self.apply:
            height, width = x.size
            new_h, new_w = int(height / 2), int(width / 2)

            block_out_mask = Image.new('RGB', (new_h, new_w), (0, 0, 0))
            original_image.paste(block_out_mask, (new_h, new_w))


        return original_image
