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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def main():
    parser = argparse.ArgumentParser()

    model_names = sorted(name for name in models.resnet.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(models.resnet.__dict__[name]))

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet50)')

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

    # 1. Load model
    # Create model
    model = models.resnet.__dict__[args.arch](pretrained=False)
    model = model.to(device)
    if not args.model_path:
        print("[Error]--model-path must be set")
        return

    if os.path.isfile(args.model_path):
        print("=> loading model '{}'".format(args.model_path))
        model_state = torch.load(args.model_path, map_location=torch.device(device))
        model.load_state_dict(model_state)
        print("=> model loaded")
    else:
        print("[Error]no model found at '{}'".format(args.model_path))
        return

    # 2. Load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    args.data = check_file(args.data)

    transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
        normalize,
    ])
    # Data loader
    data_loader, dataset = create_dataloader(args.data, 1, cache=True, transform=transform)

    # 3. Generate feature
    image_index = {
        'arch': args.arch,
        'image_size': args.image_size,
        'features': torch.Tensor(),
        'info': []
    }
    fc = FeatureExtractor(model)
    tqdm_obj = tqdm(data_loader, file=sys.stdout)
    for (input_img, target) in tqdm_obj:
        input_img = input_img.to(device)

        feature = fc.extract(input_img)
        item = dataset.get_item(tqdm_obj.n)
        image_index['features'] = torch.cat((image_index['features'], feature), 0)
        image_index['info'].append(item)

    savePath = os.path.join(args.save_dir, args.save_name)
    # Save image index
    torch.save(image_index, savePath)
    print("Image search index file generate successful!")


if __name__ == '__main__':
    main()
