import argparse
import os
import pickle
import time
import sys
import json

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
from model import Model, set_bn_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_id_from_name(img_name):
    file_name = os.path.splitext(ntpath.basename(img_name))[0]
    arr = file_name.split('_')
    return arr[0], arr[1]


def check_same_item(img1, img2):
    item_id_1, _ = get_id_from_name(img1)
    item_id_2, _ = get_id_from_name(img2)

    return item_id_1 == item_id_2

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='./save_temp/results/acm_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_model.pth', help='path of model')
    parser.add_argument('--search-idx-file', type=str, default='./save_temp/img_idx.pth', help='name of image index')
    parser.add_argument('--search-data', type=str, default='./data/train/images', help='images path of search data')
    parser.add_argument('--index-original-data', type=str, default='./data/train/images', help='images path of index origin data')

    args = parser.parse_args()

    # 1. Load image search index file
    image_index = None
    if os.path.isfile(args.search_idx_file):
        print("=> loading image search index file '{}'".format(args.search_idx_file))
        image_index = torch.load(args.search_idx_file, map_location=torch.device(device))
        print("=> image index loaded")
    else:
        print("[Error]no image index found at '{}'".format(args.search_idx_file))
        return

    image_size = 224
    idx_features = torch.as_tensor(image_index['features'])
    idx_item_info = image_index['info']

    # 2. Load model
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


    # 3. Load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        normalize,
    ])
    # Data
    _, search_dataset = create_dataloader(args.search_data, 1, cache=False, transform=transform)
    # _, origin_dataset = create_dataloader(args.index_original_data, 1, cache=False, transform=None)

    # 4. Test search result
    # Search and construct data structure
    video_score = {} # video_id => { item_id => { accumulated avg score, image accumulated avg score} }
    top_n = 5
    count = len(search_dataset)

    for i in range(count):
        if i % 1000 == 0:
            print("Now index: %d/%d" % (i, count))

        input_img, _ = search_dataset[i]
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(device)

        search_item = search_dataset.get_item(i)
        video_id, _ = get_id_from_name(search_item['img'])
        if video_id not in video_score:
            video_score[video_id] = {}

        video_record = video_score[video_id]

        # Extract feature
        with torch.no_grad():
            output = model(input_img)[0]

        feature = torch.flatten(output, 1)
        feature = feature / torch.norm(feature[0])
        feature.to(device)

        # L2 distance
        dists = torch.norm(idx_features - feature, dim=1)  # L2 distances to features
        indices = torch.argsort(dists, 0, descending=False)[:top_n]  # Top results
        scores = [(idx.item(), dists[idx].item()) for idx in indices]
        for idx, sc in scores:
            origin_item = idx_item_info[idx]
            item_id, img_sn = get_id_from_name(origin_item['img'])
            if item_id not in video_record:
                video_record[item_id] = {'avg': AverageMeter(), 'imgs':{}}
            item_record = video_record[item_id]

            item_record['avg'].update(sc)

            if img_sn not in item_record['imgs']:
                # get bbox
                item_record['imgs'][img_sn] = {'avg': AverageMeter(),
                                               'class': int(origin_item['class']),
                                               'bbox': origin_item['bbox']}

            item_record['imgs'][img_sn]['avg'].update(sc)

    print("Data construct finished!")

    class_names = ["short sleeve top", "long sleeve top", "short sleeve shirt", "long sleeve shirt", "vest top",
                   "sling top", "sleeveless top", "short outwear", "short vest", "long sleeve dress",
                   "short sleeve dress", "sleeveless dress", "long vest", "long outwear", "bodysuit", "classical",
                   "short skirt", "medium skirt", "long skirt", "shorts", "medium shorts", "trousers", "overalls"]

    sorted_video_records = {} # video_id => { "item_id":item_id, "imgs": [(img_sn, score, bbox)] }
    # Refresh data by sorting
    for video_id in video_score.keys():
        sorted_video_records[video_id] = {'item_id':0, 'result':[]}

        video_record = video_score[video_id]
        item_list = []
        for item_id in video_record.keys():
            item_list.append((item_id, video_record[item_id]['avg'].count, video_record[item_id]['avg'].avg))

        item_list.sort(key=lambda tup : (-tup[1], tup[2]))
        chosen_item_id = item_list[0][0]
        sorted_video_records[video_id]['item_id'] = chosen_item_id

        img_list = []
        for img_sn in video_record[chosen_item_id]['imgs'].keys():
            img_record = video_record[chosen_item_id]['imgs'][img_sn]
            img_list.append((img_sn, img_record['avg'].count, img_record['avg'].avg, img_record["class"], img_record['bbox']))

        img_list.sort(key=lambda tup : (-tup[1], tup[2]))
        for img_item in img_list:
            sorted_video_records[video_id]['result'].append({"img_name": img_item[0], "box": img_item[4],
                                                             "label": class_names[img_item[3]]})

    # print(sorted_video_records)
    print("Result file generated!")
    with open('submission.json', 'w') as f:
        json.dump(sorted_video_records, f)

    # Statisitcs
    total_num = len(sorted_video_records)
    accurate_num = 0
    for video_id in sorted_video_records.keys():
        if video_id == sorted_video_records[video_id]['item_id']:
            accurate_num += 1

    print("Accuracy: %0.3f%%" % (float(accurate_num * 100) / total_num))


if __name__ == '__main__':
    main()
