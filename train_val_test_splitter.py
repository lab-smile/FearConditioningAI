import os
import math
import shutil

import imageio
import pandas as pd
from PIL import Image
import numpy as np

"""
This code is made to split the IAPS dataset into train, val, test using the IAPS_image_rating_category.xlsx. 
If you want to use different excel sheet to split the data, make sure if the excel has the same format with 
IAPS_image_rating_category.xlsx 'AllSubjects_1-20 sheet.'
"""


def round_half_up(n, decimals=0):
    """
    :param n: number to round up
    :param decimals: the decimal point of round up.
    :return: rounded number

    Example: rounded_num = round_half_up(121.4, 5)
    """

    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def train_val_test_split(path='./data',
                         image_name='IAPS_Synthesized',
                         excel_name='IAPS_Synthesized.xlsx',
                         sheet_name=None,
                         pleasant_thres=6.0,
                         unpleasant_thres=4.3,
                         exclude_class=[],
                         ratio=(0.1, 0.1, 0.8),
                         random_state=4):
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    test_ratio = ratio[2]

    train_path = os.path.join(path, 'IAPS_Synthesized_train')
    val_path = os.path.join(path, 'IAPS_Synthesized_val')
    test_path = os.path.join(path, 'IAPS_Synthesized_test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    image_dir = os.path.join(path, image_name)

    excel_dir = os.path.join(path, excel_name)
    if not sheet_name:
        df = pd.read_excel(excel_dir)
    else:
        df = pd.read_excel(excel_dir, sheet_name=sheet_name)  # calling the excel file in the path.
    df = df.dropna()
    columns = df.columns

    pleasant = pd.DataFrame(columns=columns)
    unpleasant = pd.DataFrame(columns=columns)
    neutral = pd.DataFrame(columns=columns)

    if exclude_class is not None:
        for image_class in exclude_class:
            sub_df = df[df[df.columns[0]] == image_class]
            df = df.drop(sub_df.index)

    df = df.reset_index(drop=True)

    for l in range(len(df)):
        row = df.iloc[[l]]
        valence = row['valmn'][l]

        if valence <= unpleasant_thres:
            unpleasant = unpleasant.append(df.iloc[[l]])
        elif unpleasant_thres < valence < pleasant_thres:
            neutral = neutral.append(df.iloc[[l]])
        else:
            pleasant = pleasant.append(df.iloc[[l]])

    for sub_class in (unpleasant, neutral, pleasant):

        print(sub_class)
        len_sub_df = len(sub_class)
        train_n = int(round_half_up(len_sub_df * train_ratio, 0))
        val_n = int(round_half_up(len_sub_df * val_ratio, 0))
        test_n = int(round_half_up(len_sub_df * test_ratio, 0))
        test_n += 1

        #if len_sub_df % 2 == 1:
            #train_n -= 1
            #test_n -= 1

        train_sample = sub_class.sample(n=train_n, replace=False, random_state=random_state)
        sub_class = sub_class.drop(train_sample.index)
        val_sample = sub_class.sample(n=val_n, replace=False, random_state=random_state)
        sub_class = sub_class.drop(val_sample.index)
        test_sample = sub_class.sample(n=test_n, replace=False, random_state=random_state)

        for jpg in train_sample['IAPS']:
            jpg_path = os.path.join(image_dir, str(jpg).replace('.0', '') + '.jpg')
            shutil.copy(jpg_path, train_path)

        for jpg in val_sample['IAPS']:
            jpg_path = os.path.join(image_dir, str(jpg).replace('.0', '') + '.jpg')
            shutil.copy(jpg_path, val_path)

        for jpg in test_sample['IAPS']:
            jpg_path = os.path.join(image_dir, str(jpg).replace('.0', '') + '.jpg')
            shutil.copy(jpg_path, test_path)


def class_extract(class_name='EroticMale',
                  path='/data/seowung/data',
                  image_name='IAPS2',
                  excel_name='IAPS_image_rating_category.xlsx',
                  sheet_name='AllSubjects_1-20'):

    print("The Function 'Class Extract' is Running")
    print("Extracting Class : ", class_name)

    image_dir = os.path.join(path, image_name)

    image_path = os.path.join(path, class_name)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    excel_dir = os.path.join(path, excel_name)
    df = pd.read_excel(excel_dir, sheet_name=sheet_name)  # calling the excel file in the path.
    sub_df = df[df[df.columns[0]] == class_name]

    sub_df = sub_df.reset_index(drop=True)

    for jpg in sub_df['IAPS']:
        jpg_path = os.path.join(image_dir, str(jpg).replace('.0', '') + '.jpg')
        shutil.copy(jpg_path, image_path)


def index_extract(input_xlsx='IAPS_Conditioning_Unpleasant2.xlsx', path = '/data/seowung/data', image_name='IAPS2'):
    print("The Function 'Index Extract' is Running")
    print("Image list : ", input_xlsx)

    image_dir = os.path.join(path, image_name)

    image_path = os.path.join(path, input_xlsx[:-5])
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    excel_dir = os.path.join(path, input_xlsx)
    df = pd.read_excel(excel_dir)
    df = df.dropna()
    df = df.reset_index(drop=True)

    for jpg in df['IAPS']:
        jpg_path = os.path.join(image_dir, str(jpg).replace('.0', '') + '.jpg')
        shutil.copy(jpg_path, image_path)




def main():
    train_val_test_split()
    #class_extract()
    #Conditioning_Dataset_Creator(path='/data/seowung/data',
    #                             image_list=['IAPS_Conditioning_Finetune_Erotic', 'IAPS_Conditioning_Finetune_Mutilation'],
    #                             gabor_path = 'gabor_patch',
    #                             orientation = [45, 135])
    #index_extract()


if __name__ == '__main__':
    main()
