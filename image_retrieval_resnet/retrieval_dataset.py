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
from utils.utils import *
import ntpath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def check_same_item(img1, img2):
    file_name1 = ntpath.basename(img1)
    file_name2 = ntpath.basename(img2)

    arr = file_name1.split('_')
    item_id_1 = arr[0]
    arr = file_name2.split('_')
    item_id_2 = arr[0]

    return item_id_1 == item_id_2

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, default='./save_temp/resnet50.pth', help='path of model')
    parser.add_argument('--search-idx-file', type=str, default='./img_index/img_idx.pth', help='name of image index')
    parser.add_argument('--search-data', type=str, default='./data/train/images', help='images path of search data')
    parser.add_argument('--index-original-data', type=str, default='./data/train/images', help='images path of index origin data')

    args = parser.parse_args()

    # 1. Load image search index file
    image_index = None
    if os.path.isfile(args.search_idx_file):
        print("=> loading image search index file '{}'".format(args.search_idx_file))
        image_index = torch.load(args.search_idx_file)
        print("=> image index loaded")
    else:
        print("[Error]no image index found at '{}'".format(args.search_idx_file))
        return

    image_size = image_index['image_size']
    idx_features = torch.as_tensor(image_index['features'])

    # 2. Load model
    # Create model
    model = models.resnet.__dict__[image_index['arch']](pretrained=False)
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

    # 3. Load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    args.search_data = check_file(args.search_data)
    args.index_original_data = check_file(args.index_original_data)

    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        normalize,
    ])
    # Data
    _, search_dataset = create_dataloader(args.search_data, 1, cache=True, transform=transform)
    _, origin_dataset = create_dataloader(args.index_original_data, 1, cache=True, transform=None)

    # 4. Test search result
    fc = FeatureExtractor(model)

    total_num = 0
    accurate_num = 0
    for i in range(0, len(search_dataset)):
        input_img, _ = search_dataset[i]
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(device)

        feature = fc.extract(input_img)

        top_n = 5
        dists = np.linalg.norm(idx_features - feature, axis=1)  # L2 distances to features
        indices = np.argsort(dists)[:top_n]  # Top results
        scores = [(idx, dists[idx]) for idx in indices]
        search_item = search_dataset.get_item(i)
        origin_item = origin_dataset.get_item(indices[0])
        print("Search: {}, got: {}".format(search_item['img'], origin_item['img']))

        total_num += 1
        if check_same_item(search_item['img'], origin_item['img']):
            accurate_num += 1

        print("Accuracy: %0.3f%%" % (float(accurate_num * 100) / total_num))


if __name__ == '__main__':
    main()
