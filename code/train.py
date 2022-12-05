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

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights='IMAGENET1K_V1')
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.weight = nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)    
    model.fc.bias = nn.init.zeros_(model.fc.bias)

    return model
        
def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    
    model.to(device)
    # Load the datasets
    train_data, val_data = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=True, drop_last = False)


    # Initialize the optimizer (Adam) to train the last layer of the model.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    best_acc = -1
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train_losses_epoch, val_losses_epoch = [], []
        # train
        model.train()
        for (X_train, y_train) in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred_train = model(X_train)
            loss = criterion(y_pred_train, y_train)
            train_losses_epoch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = evaluate_model(model, train_loader, device) 
        # train_accs.append(train_acc)
        train_loss = np.mean(train_losses_epoch)
        # train_losses.append(train_loss)
        print(f'Train accuracy: {train_acc*100:.2f}% | Training loss: {train_loss:.4f}')

        # validate
        model.eval()
        with torch.no_grad():
            for (X_val, y_val) in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                y_pred_val = model(X_val)
                loss = criterion(y_pred_val, y_val)
                val_losses_epoch.append(loss.item())

        val_acc = evaluate_model(model, val_loader, device)
        # val_accs.append(val_acc)
        val_loss = np.mean(val_losses_epoch)
        # val_losses.append(val_loss)
        print(f'Validation accuracy: {val_acc*100:.2f}% | Validation loss: {val_loss:.4f}')
        # check if the model is the best so far and save it
        if val_acc > best_acc:
            best_model_state = model.state_dict()    # save best model state 
            best_acc = val_acc

    # Load the best model on val accuracy and return it.
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, checkpoint_name)

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    model.to(device)
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    batch_sizes, accs = [], []
    # calculate the metrics for each batch
    with torch.no_grad():
        for X_test, y_test in data_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            preds = model(X_test)
            batch_sizes.append(X_test.shape[0])
            acc = torch.mean((torch.argmax(preds, dim=1) == y_test).float()).item()
            accs.append(acc)
    # calculate the average accuracy for the whole dataset, weighted by batch size
    accuracy = np.average(accs, weights=batch_sizes)   

    return accuracy


def main(lr, batch_size, epochs, data_dir, model_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        model_dir: Directory where the model should be saved.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(torch.cuda.is_available()):
        print("Using GPU: " + torch.cuda.get_device_name(device))
    else:
        print("Using CPU")
     
    # Load the model
    model = get_model()
    model.to(device)
    
    # Train the model
    filename = f'resnet_lr={lr}_batchsize={batch_size}_epochs={epochs}_augmentations={augmentation_name}.pt'
    checkpoint_name = os.path.join(model_dir, filename)
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name)

    # Evaluate the model on the test set
    test_data = get_test_set(data_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, drop_last = False)
    test_acc = evaluate_model(model, test_loader, device)
    print(f'Test accuracy: {test_acc*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument(
        "--model_dir", type=str, default="./save/models", help="path to save models"
    )
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    
    kwargs = vars(args)
    main(**kwargs)
