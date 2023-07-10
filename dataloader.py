from __future__ import print_function, division
# base libraries
import os

# libraries for arithmetic
import numpy as np
import pandas as pd
import random
from skimage import io
from utils import GaussianBlur, Quadrant_Processing, Quadrant_Processing_Conditioning

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler

import PIL
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# This part of code is not used currently

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def data_transform(train_folder, val_folder, test_folder):
    r"""This part of the code preprocess the images in train, val, test folders.
    The preprocessing includes resizing, random rotation, color jittering, gaussian blur, random horizontal flip,
    and normalization. You can apply Quadrant processing before the normalization if you want.

    Args:
         train_folder: the directory of training images of the dataset
         val_folder: the directory of validation images of the dataset
         test_folder: the directory of validation images of the dataset
         is_gist: if you set to True, it will apply gist transformation
         is_saliency: if you set to True it will apply saliency transformation

    Example:
        data_transforms = data_transform(train_folder, val_folder, test_folder)
    """
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        train_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        val_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        test_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    }
    return data_transforms


def data_transform_conditioning(train_folder1, train_folder2, val_folder, test_folder, gabor_dir1=None,
                                 gabor_dir2=None, manipulation=0.0):
    r"""This part of the code preprocess the images in train, val, test folders.
    The preprocessing includes resizing, random rotation, color jittering, gaussian blur, random horizontal flip,
    and normalization. You can apply Quadrant processing before the normalization if you want.

    Args:
         train_folder: the directory of training images of the dataset
         val_folder: the directory of validation images of the dataset
         test_folder: the directory of validation images of the dataset
         is_gist: if you set to True, it will apply gist transformation
         is_saliency: if you set to True it will apply saliency transformation
         gabor_dir: the directory where gabor_patch is saved

    Example:
        data_transforms = data_transform(train_folder, val_folder, test_folder)
        :param manipulation:

    """

    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        train_folder1: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            Quadrant_Processing_Conditioning(4, gabor_dir1),
            transforms.ToTensor(),
            normalize,
        ]),
        train_folder2: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            Quadrant_Processing_Conditioning(4, gabor_dir2),
            transforms.ToTensor(),
            normalize,
        ]),
        val_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            Quadrant_Processing(4),
            transforms.ToTensor(),
            normalize,
        ]),
        test_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.CenterCrop(image_size),
            Quadrant_Processing(4),
            transforms.ToTensor(),
            normalize,
        ])
    }

    return data_transforms


def data_transform_quadrant_finetune(train_folder, val_folder, test_folder, is_gist=False, is_saliency=False):
    r"""This part of the code preprocess the images in train, val, test folders.
    The preprocessing includes resizing, random rotation, color jittering, gaussian blur, random horizontal flip,
    and normalization. You can apply Quadrant processing before the normalization if you want.

    Args:
         train_folder: the directory of training images of the dataset
         val_folder: the directory of validation images of the dataset
         test_folder: the directory of validation images of the dataset
         is_gist: if you set to True, it will apply gist transformation
         is_saliency: if you set to True it will apply saliency transformation

    Example:
        data_transforms = data_transform(train_folder, val_folder, test_folder)
    """
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        train_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            Quadrant_Processing(4),
            transforms.ToTensor(),
            normalize,
        ]),
        val_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)], p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            Quadrant_Processing(4),
            transforms.ToTensor(),
            normalize,
        ]),
        test_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.CenterCrop(image_size),
            Quadrant_Processing(4),
            transforms.ToTensor(),
            normalize,
        ])
    }

    if is_gist:
        # this transform is not changing the image self so it can be used in test as well.
        data_transforms['gist'] = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            normalize,
            transforms.ToPILImage()
        ])
    if is_saliency:
        # this transform is not changing the image self so it can be used in test as well.
        data_transforms['saliency'] = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            normalize,
            transforms.ToPILImage()
        ])
    return data_transforms


def data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test):
    csv = {
        train_folder: csv_train,
        val_folder: csv_val,
        test_folder: csv_test
    }
    return csv


def data_csv2(train_folder1, train_folder2, val_folder, test_folder, csv_train1, csv_train2, csv_val, csv_test):
    csv = {
        train_folder1: csv_train1,
        train_folder2: csv_train2,
        val_folder: csv_val,
        test_folder: csv_test
    }
    return csv


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


# Classification dataloader : the name of the folder is the class name
def cls_dataloader_tuning(data_dir, test_folder, batch_size, category):
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(os.path.join(data_dir, test_folder), transform=data_transform)

    idx = get_indices(dataset, dataset.class_to_idx[category])
    sampler = SubsetRandomSampler(idx)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=32, pin_memory=True, sampler=sampler)

    return loader, idx, len(idx)


def cls_dataloader(data_dir, train_folder, val_folder, test_folder, batch_size, istrain):
    data_transforms = data_transform(train_folder, val_folder, test_folder)

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in ([train_folder, val_folder] if istrain else [test_folder])
    }

    if istrain:
        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                num_workers=16,
                pin_memory=True,
                # sampler=torch.utils.data.RandomSampler(image_datasets[x], replacement=False),
                sampler=ImbalancedDatasetSampler(image_datasets[x]),
            )
            for x in [train_folder, val_folder]

        }
    else:

        dataloaders = {
            test_folder: DataLoader(
                image_datasets[test_folder],
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True
            )
        }

    dataset_sizes = {x: len(image_datasets[x]) for x in ([train_folder, val_folder] if istrain else [test_folder])}

    for x in ([train_folder, val_folder] if istrain else [test_folder]):
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    print("Classes: ")
    if istrain:
        class_names = image_datasets[train_folder].classes
        class_to_idx = image_datasets[train_folder].class_to_idx
    else:
        class_names = image_datasets[test_folder].classes
        class_to_idx = image_datasets[test_folder].class_to_idx
    print(class_names)
    return dataloaders, dataset_sizes, class_names, class_to_idx  # , train_folder, val_folder, test_folder


# regression dataloader
def reg_dataloader(data_dir, train_folder, val_folder, test_folder, csv_train, csv_val, csv_test,
                   batch_size, istrain, seed=0):
    """
           Args:
              istrain: is training or not
           """

    data_transforms = data_transform(train_folder, val_folder, test_folder)

    csv_files = data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test)

    image_datasets = {
        x: RegressionDataset(
            csv_file=csv_files[x],
            root_dir=os.path.join(data_dir, x),
            transform=data_transforms[x],
            istrain=istrain,
            seed=seed
        )
        for x in ([train_folder, val_folder] if istrain else [test_folder])
    }

    if istrain:
        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                num_workers=24,
                pin_memory=True,
                sampler=ImbalancedDatasetSampler(image_datasets[x], callback_get_label=reg_callback_get_label),
            )
            for x in [train_folder, val_folder]

        }
    else:
        dataloaders = {
            test_folder: DataLoader(
                image_datasets[test_folder],
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True
            )
        }

    dataset_sizes = {x: len(image_datasets[x]) for x in ([train_folder, val_folder] if istrain else [test_folder])}

    for x in ([train_folder, val_folder] if istrain else [test_folder]):
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    return dataloaders, dataset_sizes


# conditioning dataloader
def cond_dataloader(data_dir, train_folder1, train_folder2, val_folder, test_folder, csv_train1, csv_train2, csv_val,
                     csv_test,batch_size, istrain, gabor_dir1=None, gabor_dir2=None, manipulation=0.0, seed=0):
    r"""This part of the code has same structure with reg_dataloader. The only difference is the applied transform to
        the dataset. Unlike reg_dataloader, the cond_dataloader has Quadrant processing module for 0-delay conditioning.

        Args:
             data_dir: the base directory where the datasets are.
             train_folder: the name of the training dataset folder.
             val_folder: the name of the validation dataset folder.
             test_folder: the name of the test dataset folder.
             csv_train: the directory of csv files where valence of training dataset is saved.
             csv_val: the directory of csv files where valence of validation dataset is saved.
             csv_test: the directory of csv files where valence of test dataset is saved.
             batch_size: the batch size you will feed the model for single iteration.
             istrain: boolean input. If 'True', the dataloader will process the training and validation dataset.
             is_gist: boolean input. True if you want to apply gist transformation
             is_saliency: boolean input. True if you want to apply saliency transformation.
             gabor_dir:

        Example:
            dataloaders_train, dataset_sizes_train = reg_dataloader(args.data_dir,
                                                            args.TRAIN, args.VAL, args.TEST, args.csv_train,
                                                            args.csv_val, args.csv_test, args.batch_size, istrain=True,
                                                            is_gist=is_gist, is_saliency=is_saliency)
        """

    data_transforms = data_transform_conditioning(train_folder1, train_folder2, val_folder, test_folder, gabor_dir1,
                                                   gabor_dir2, manipulation)

    csv_files = data_csv2(train_folder1, train_folder2, val_folder, test_folder, csv_train1, csv_train2, csv_val,
                          csv_test)

    image_datasets = {
        x: RegressionDataset(
            csv_file=csv_files[x],
            root_dir=os.path.join(data_dir, x),
            transform=data_transforms[x],
            istrain=istrain,
            seed=seed
        )
        for x in ([train_folder1, train_folder2, val_folder] if istrain else [test_folder])
    }
    if istrain:
        train_folder = 'IAPS_Conditioning_Finetune'
        train = torch.utils.data.ConcatDataset([image_datasets[train_folder1], image_datasets[train_folder2]])
        image_datasets = {train_folder: train,
                          val_folder: image_datasets[val_folder]}

    if istrain:
        if manipulation == 0.0:
            dataloaders = {
                x: DataLoader(
                    image_datasets[x],
                    batch_size=batch_size,
                    num_workers=24,
                    pin_memory=True,
                    sampler=ImbalancedDatasetSampler(image_datasets[x], callback_get_label=reg_callback_get_label),
                    # sampler=torch.utils.data.RandomSampler(image_datasets[x], replacement=False),
                )
                for x in [train_folder, val_folder]
            }
        else:
            dataloaders = {
                x: DataLoader(
                    image_datasets[x],
                    batch_size=batch_size,
                    pin_memory=True,
                    sampler=ImbalancedDatasetSampler(image_datasets[x], callback_get_label=reg_callback_get_label),
                    # sampler=torch.utils.data.RandomSampler(image_datasets[x], replacement=False),
                )
                for x in [train_folder, val_folder]
            }
    else:
        dataloaders = {
            test_folder: DataLoader(
                image_datasets[test_folder],
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True
            )
        }

    dataset_sizes = {x: len(image_datasets[x]) for x in ([train_folder, val_folder] if istrain else [test_folder])}

    for x in ([train_folder, val_folder] if istrain else [test_folder]):
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    return dataloaders, dataset_sizes


# quadrant finetune dataloader
def quadrant_finetune_dataloader(data_dir, train_folder, val_folder, test_folder, csv_train, csv_val, csv_test,
                                 batch_size, istrain, seed=0):
    r"""This part of the code has same structure with reg_dataloader. The only difference is the applied transform to
        the dataset. Unlike reg_dataloader, the cond_dataloader has Quadrant processing module for 0-delay conditioning.

        Args:
             data_dir: the base directory where the datasets are.
             train_folder: the name of the training dataset folder.
             val_folder: the name of the validation dataset folder.
             test_folder: the name of the test dataset folder.
             csv_train: the directory of csv files where valence of training dataset is saved.
             csv_val: the directory of csv files where valence of validation dataset is saved.
             csv_test: the directory of csv files where valence of test dataset is saved.
             batch_size: the batch size you will feed the model for single iteration.
             istrain: boolean input. If 'True', the dataloader will process the training and validation dataset.
             is_gist: boolean input. True if you want to apply gist transformation
             is_saliency: boolean input. True if you want to apply saliency transformation.

        Example:
            dataloaders_train, dataset_sizes_train = reg_dataloader(args.data_dir,
                                                            args.TRAIN, args.VAL, args.TEST, args.csv_train,
                                                            args.csv_val, args.csv_test, args.batch_size, istrain=True,
                                                            is_gist=is_gist, is_saliency=is_saliency)
        """

    data_transforms = data_transform_quadrant_finetune(train_folder, val_folder, test_folder)

    csv_files = data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test)

    image_datasets = {
        x: RegressionDataset(
            csv_file=csv_files[x],
            root_dir=os.path.join(data_dir, x),
            transform=data_transforms[x],
            istrain=istrain,
            seed=seed
        )
        for x in ([train_folder, val_folder] if istrain else [test_folder])
    }

    if istrain:
        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                num_workers=24,
                pin_memory=True,
                sampler=ImbalancedDatasetSampler(image_datasets[x], callback_get_label=reg_callback_get_label),
            )
            for x in [train_folder, val_folder]

        }
    else:
        dataloaders = {
            test_folder: DataLoader(
                image_datasets[test_folder],
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True
            )
        }

    dataset_sizes = {x: len(image_datasets[x]) for x in ([train_folder, val_folder] if istrain else [test_folder])}

    for x in ([train_folder, val_folder] if istrain else [test_folder]):
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    return dataloaders, dataset_sizes


def image_patch_loader(transform, dir):
    """load image, returns cuda tensor"""

    image = Image.open(dir)
    image = transform(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


# The label is read from csv
class RegressionDataset(Dataset):
    """dataset only used for Regression ."""

    def __init__(self, csv_file, root_dir, transform=None, istrain=True, seed=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            is_gist: whether use gist features as the input of low-road
        """
        self.label_csv = pd.read_csv(csv_file, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform
        self.istrain = istrain
        self.seed = seed


    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #np.random.seed(self.seed)
        #random.seed(self.seed)
        #torch.manual_seed(self.seed)
        #os.environ["PYTHONHASHSEED"] = str(self.seed)

        name = self.label_csv.iloc[idx, 0]
        if isinstance(name, int):
            name = str(name) + '.' + self.label_csv.iloc[idx, 1]
        elif isinstance(name, float):
            name = str(int(name) if int(name) == name else name)
            name = name + '.' + self.label_csv.iloc[idx, 1]
        else:
            name = str(name) + '.' + self.label_csv.iloc[idx, 1]

        img_name = os.path.join(self.root_dir, name)

        image = io.imread(img_name)
        if len(image.shape) == 3:  # RGB
            image = transforms.ToPILImage()(image)
        else:  # grayscale
            image = image.reshape(1, image.shape[0], image.shape[1])
            image = transforms.ToPILImage()(image)

        label = self.label_csv.iloc[idx, 3]

        if self.istrain:
            noise = np.random.normal(loc=0, scale=0.05, size=1)
            label = float(label) + float(noise)

        if self.transform:  # the transform is generated based on training or test datasets; if it is testing, the transform is not changing the image
            # print(name)
            if image.mode == 'RGBA':
                image.load()
                background = PIL.Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            output_image = self.transform(image)
        else:
            output_image = image

        return name, output_image, float(label)


def reg_callback_get_label(dataset, idx):
    return round(dataset[idx][2])
