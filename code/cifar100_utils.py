################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import torch

from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
from torchvision import transforms


def add_augmentation(augmentation_name, transform_list):
    """
    Adds an augmentation transform to the list.
    Args:
        augmentation_name: Name of the augmentation to use.
        transform_list: List of transforms to add the augmentation to.

    """
    # Create a new transformation based on the augmentation_name and add it to the transform_list
    if augmentation_name == 'HorizontalFlip':
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    elif augmentation_name == 'VerticalFlip':
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    elif augmentation_name == 'RandomRotation':
        transform_list.append(transforms.RandomRotation(45))
    elif augmentation_name == 'RandomCrop':
        transform_list.append(transforms.RandomCrop(32, padding=4))
    elif augmentation_name == 'ColorJitter':
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    elif augmentation_name == 'RandomErasing':
        transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))
    elif augmentation_name == 'GaussianBlur':
        transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
    elif augmentation_name == 'RandomGrayscale':
        transform_list.append(transforms.RandomGrayscale(p=0.2))
    elif augmentation_name == 'RandomPerspective':
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0))
    elif augmentation_name == 'RandomAffine':
        transform_list.append(transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=45, resample=False, fillcolor=0))
    elif augmentation_name == 'RandomResizedCrop':
        transform_list.append(transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=2))
    elif augmentation_name == 'CenterCrop':
        transform_list.append(transforms.CenterCrop(32))
    elif augmentation_name == 'FiveCrop':
        transform_list.append(transforms.FiveCrop(32))
    elif augmentation_name == 'None':
        pass
    else:
        raise ValueError('Augmentation name not recognized.')


def get_train_validation_set(data_dir, validation_size=5000, augmentation_name=None):
    """
    Returns the training and validation set of CIFAR100.

    Args:
        data_dir: Directory where the data should be stored.
        validation_size: Size of the validation size
        augmentation_name: The name of the augmentation to use.

    Returns:
        train_dataset: Training dataset of CIFAR100
        val_dataset: Validation dataset of CIFAR100
    """

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = [transforms.Resize((224, 224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)]
    if augmentation_name is not None:
        add_augmentation(augmentation_name, train_transform)
    train_transform = transforms.Compose(train_transform)

    val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    # We need to load the dataset twice because we want to use them with different transformations
    train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=val_transform)

    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_dataset):
        raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
            len(train_dataset), validation_size))

    train_dataset, _ = random_split(train_dataset,
                                    lengths=[len(train_dataset) - validation_size, validation_size],
                                    generator=torch.Generator().manual_seed(42))
    _, val_dataset = random_split(val_dataset,
                                  lengths=[len(val_dataset) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        # TODO: Given a batch of images, add Gaussian noise to each image.

        img += torch.normal(self.mean, self.std, size=img.shape)

        return img
        
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def get_test_set(data_dir, add_noise=False):
    """
    Returns the test dataset of CIFAR100.

    Args:
        data_dir: Directory where the data should be stored
    Returns:
        test_dataset: The test dataset of CIFAR100.
    """

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    if add_noise:
        test_transform = transforms.Compose([test_transform, AddGaussianNoise()])
    test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    return test_dataset
