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

    parser.add_argument('--model-path', type=str, default='./save_temp/resnet50.pth', help='path of model')
    parser.add_argument('--pretrained', dest='pretrained', default=True, help='use pre-trained model')
    parser.add_argument('--search-idx-file', type=str, default='./img_index/img_idx.pth', help='name of image index')
    parser.add_argument('--search-data', type=str, default='./data/train/images', help='images path of search data')
    parser.add_argument('--index-original-data', type=str, default='./data/train/images',
                        help='images path of index origin data')
    parser.add_argument('--idx', type=int, default=0, help='index of query image')

    args = parser.parse_args()

    # 1. Load image search index file
    image_index = None
    if os.path.isfile(args.search_idx_file):
        print("=> loading image search index file '{}'".format(args.search_idx_file))
        image_index = torch.load(args.search_idx_file, map_location=torch.device('cpu'))
        print("=> image index loaded")
    else:
        print("[Error]no image index found at '{}'".format(args.search_idx_file))
        return

    image_size = image_index['image_size']
    idx_features = torch.as_tensor(image_index['features'])

    # 2. Load model
    # Create model
    model = models.resnet.__dict__[args.arch](pretrained=args.pretrained)
    if not args.pretrained:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23)

    model = model.to(device)
    if not args.model_path:
        print("[Error]--model-path must be set")
        return

    if not args.pretrained:
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

    test_idx = args.idx
    input_img, _ = search_dataset[test_idx]
    input_img = input_img.unsqueeze(0)
    input_img = input_img.to(device)

    feature = fc.extract(input_img)
    top_n = 10
    # L2 distance
    dists = np.linalg.norm(idx_features - feature.cpu(), axis=1)  # L2 distances to features
    print(dists)
    indices = np.argsort(dists)[:top_n]  # Top results
    scores = [(idx, dists[idx]) for idx in indices]

    # L1 distance
    # dists = torch.sum(torch.abs(idx_features - feature.cpu()), 1)  # L2 distances to features
    # print(dists)
    # indices = torch.argsort(dists, 0, descending=False)[:top_n]  # Top results
    # scores = [(idx, dists[idx]) for idx in indices]

    # cosine similarity
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # similarity = cos(idx_features, feature.cpu())
    # print(similarity)
    # indices = torch.argsort(similarity, 0, descending=True)[:top_n]  # Top results
    # scores = [(idx, similarity[idx]) for idx in indices]
    print(scores)

    # 5. Show result

    figure = plt.figure(figsize=(8, 4))
    plt.rcParams.update({'axes.titlesize': 'small'})
    # Query image
    query_image = search_dataset.get_image(test_idx)
    figure.add_subplot(2, top_n, 1)
    plt.title("Query image")
    plt.axis("off")
    plt.imshow(query_image)

    # Results
    for i in range(0, len(scores)):
        idx, sc = scores[i]
        result_image = origin_dataset.get_image(idx)
        figure.add_subplot(2, top_n, top_n + 1 + i)
        # plt.title("Top %d \n score: %0.5f" % (i + 1, sc))
        plt.title("Top %d" % (i + 1))
        plt.axis("off")
        plt.imshow(result_image)

    plt.show()


if __name__ == '__main__':
    main()
