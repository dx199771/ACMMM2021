import argparse
import os
import pickle
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm
from typing import Type, Any, Callable, Union, List, Optional
from utils.general import check_file
from utils.datasets import create_dataloader
from utils.feature_extractor import FeatureExtractor
from model import Model, set_bn_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()

    model_names = sorted(name for name in models.resnet.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(models.resnet.__dict__[name]))

    parser.add_argument('--append', dest='append', default=False, help='append new index')
    parser.add_argument('--image-size', type=int, default=224, help='resize image to ')
    parser.add_argument('--model-path', type=str, default='', help='path of model')
    parser.add_argument('--data', type=str, default='', help='images path of data')
    parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the index file',
                        default='save_temp', type=str)
    parser.add_argument('--save-name', type=str, default='img_idx.pth', help='name of image index')

    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    savePath = os.path.join(args.save_dir, args.save_name)

    image_index = {
        'arch': 'cgd',
        'image_size': args.image_size,
        'features': torch.Tensor().to(device),
        'info': []
    }

    if args.append:
        # 1. Load image search index file
        if os.path.isfile(savePath):
            print("=> loading image search index file '{}'".format(savePath))
            image_index = torch.load(savePath, map_location=torch.device(device))
            print("=> image index loaded")
        else:
            print("[Warning]Image index does not exist at '{}'. Create new!".format(savePath))


    # 1. Load model
    # Create model
    model = Model('resnet50', 'SG', 1536, num_classes=23).to(device)
    model.eval()
    if not args.model_path:
        print("[Error]--model-path must be set")
        return

    if os.path.isfile(args.model_path):
        print("=> loading model '{}'".format(args.model_path))
        model_state = torch.load(args.model_path, map_location=torch.device(device))
        model.load_state_dict(model_state, strict=False)
        print("=> model loaded")
    else:
        print("[Error]no model found at '{}'".format(args.model_path))
        return

    # 2. Load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
        normalize,
    ])
    # Data loader
    data_loader, dataset = create_dataloader(args.data, 1, cache=False, transform=transform)

    # 3. Generate feature
    count = len(dataset)

    for i in range(count):
        if i % 1000 == 0:
            print("Now index: %d/%d" % (i, count))
        input_img, _ = dataset[i]
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(device)
        with torch.no_grad():
            feature = model(input_img)[0]
            item = dataset.get_item(i)
            image_index['features'] = torch.cat((image_index['features'], feature), 0)
            image_index['info'].append(item)

    # Save image index
    torch.save(image_index, savePath)
    print("Image search index file generate successful!")


if __name__ == '__main__':
    main()
