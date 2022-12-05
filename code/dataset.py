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

"""Defines helper functions for loading data and constructing dataloaders."""
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, random_split
import torch

DATASET = {"cifar10": CIFAR10, "cifar100": CIFAR100}
        

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


def load_dataset(args):
    
    test_transform = AddGaussianNoise() if args.test_noise else None
    train_dataset = DATASET[args.dataset](
        args.data_dir, transform=None, download=True, train=True
    )

    ratio = 0.2
    valid_size = int(len(train_dataset) * ratio)
    train_size = int(len(train_dataset) - valid_size)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, valid_size])

    test_dataset = DATASET[args.dataset](
        args.data_dir, transform=test_transform, download=True, train=False
    )

    return train_dataset, val_dataset, test_dataset


def construct_dataloader(args, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
