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

"""Helper script to evaluate noise robustness."""
import os
import argparse
import torch
from train import set_seed, get_model, evaluate_model
from cifar100_utils import get_test_set
# from dataset import load_dataset, construct_dataloader


def main(args):
    
    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(torch.cuda.is_available()):
        print("Using GPU: " + torch.cuda.get_device_name(device))
    else:
        print("Using CPU")
        
    # Set the seed for reproducibility
    set_seed(args.seed)
     
    # Load the model
    model = get_model()
    model.load_state_dict(torch.load(args.checkpoint_name))
    model.to(device)

    # Evaluate the model on the test set
    test_data = get_test_set(args.data_dir, args.test_noise)
    test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True, drop_last = False)
    # test_data = load_dataset(args)
    # test_loader = construct_dataloader(args, test_data)
    test_acc = evaluate_model(model, test_loader, device)
    print(f'Test accuracy: {test_acc*100:.2f}%')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--batch_size', default=128, type=int, help='Minibatch size')
    # parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    # parser.add_argument(
    #     "--model_dir", type=str, default="./save/models", help="path to save models"
    # )
    parser.add_argument(
        "--checkpoint_name", type=str, default=None, help="path to load best trained model"
    )
    parser.add_argument(
        "--evaluate", default=False, action="store_true", help="evaluate model test set"
    )
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="whether to add noise to the test images",
    )
    
    args = parser.parse_args()
    # kwargs = vars(args)
    print(args)
    
    if not args.checkpoint_name:
        raise ValueError("Please specify a checkpoint name to load!")
    elif not os.path.exists(args.checkpoint_name):
        raise ValueError("Checkpoint file does not exist!")
    
    if args.evaluate:
        if args.test_noise:
            print("Model evaluation w/ random noise\n")
        else:
            print("Model evaluation w/o random noise\n")
        main(args)
    else:
        raise ValueError("Enable flag --evaluate!")
