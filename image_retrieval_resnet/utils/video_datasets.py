import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import queue
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.general import torch_distributed_zero_first, xywh2xyxy

video_formats = ['.mp4', '.avi']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(file_path):
    video = cv2.VideoCapture(file_path)
    # Returns exif-corrected PIL size
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    # frames = video.get(cv2.CAP_PROP_FRAME_COUNT)


    video.release()
    return (height, width)


def create_dataloader(path, batch_size, cache=True, transform=None, sampler=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    dataset = LoadVideosAndLabels(path, transform=transform, cache_images=cache)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 1, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=nw,
                                             pin_memory=True)
    return dataloader, dataset


def create_dataloader_with_dataset(dataset, batch_size, sampler=None):
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 1, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=nw,
                                             pin_memory=True)
    return dataloader


class LoadVideosAndLabels(Dataset):  # for training/testing
    cached_queue = queue.Queue()
    cached_video = {}
    max_queue = 100


    def __init__(self, path, transform=None, cache_images=False):
        try:
            f = []  # video files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                elif os.path.isdir(p):  # folder
                    f += glob.iglob(p + os.sep + '*.*')
                else:
                    raise Exception('%s does not exist' % p)
            self.video_files = sorted(
                [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in video_formats])
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))
        video_number = len(self.video_files)
        assert video_number > 0, 'No images found in %s' % (path)

        # Define labels
        self.label_files = [x.replace('videos', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.video_files]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.video_files):  # dataset changed
                cache = self.cache_data(cache_path)  # re-cache
        else:
            cache = self.cache_data(cache_path)  # cache

        # data stores all the parameters of all the samples
        self.data = cache['data']

        self.labels = []
        for i in range(0, len(self.data)):
            self.labels.append(int(self.data[i]['label'][1]))

        self.class_to_idx = cache['class_to_idx']

        self.n = len(self.data)  # number of samples
        self.cache_images = cache_images
        self.images = [None] * self.n
        self.transform = transform

    def cache_data(self, path='data.cache'):
        # Cache dataset labels, check images and read shapes
        x = {"data": [], "class_to_idx": {}}  # dict
        pbar = tqdm(zip(self.video_files, self.label_files), desc='Loading labels', total=len(self.video_files))
        now_sn = 0
        for (video, label) in pbar:
            l = []
            shape = exif_size(video)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), 'video size <10 pixels'
            if os.path.isfile(label):
                with open(label, 'r') as f:
                    origin_l = [x.split() for x in f.read().splitlines()]

                    for line in origin_l:
                        if line[1] not in x['class_to_idx']:
                            x['class_to_idx'][line[1]] = int(line[1])

                        # Substitute with index
                        line[1] = x['class_to_idx'][line[1]]

                    l = np.array(origin_l, dtype=np.float32)  # labels

            if len(l) == 0:
                l = np.zeros((0, 6), dtype=np.float32)

            for one_l in l:
                x["data"].append({'video': video, 'shape': shape, 'label': one_l})

        x['hash'] = get_hash(self.label_files + self.video_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return self.n

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        item = self.data[index]

        img = None
        if self.cache_images and self.images[index] is not None:
            img = self.images[index]
        else:
            try:
                img = self.load_image(index)
                # img = Image.fromarray(img)
            except Exception as e:
                print(item)
                print(e)

            if self.transform is not None:
                img = self.transform(img)

            if self.cache_images:
                self.images[index] = img

        return img, self.labels[index]

    def get_item(self, index):
        return self.data[index]

    def get_image(self, index):
        img = self.load_image(index)

        return img

    def load_image(self, index):
        # loads 1 image from dataset, returns clipped img
        item = self.data[index]
        if item["video"] in self.cached_video:
            video = self.cached_video[item["video"]]
        else:
            video = cv2.VideoCapture(item["video"])
            self.cache_video(item["video"], video)

        # 0 based frame number
        video.set(cv2.CAP_PROP_POS_FRAMES, int(item["label"][0] - 1))

        ret, img = video.read()
        if ret == True:
            # Save the resulting frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print("Read video frame failed: " + item["video"] + ", frame: " + str(item["label"][0] - 1))
            return None

        img = np.array(img)
        h, w = item['shape']
        b = item['label'][2:] * [w, h, w, h]  # box
        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
        return_img = Image.fromarray(img[b[1]:b[3], b[0]:b[2]])

        return return_img

    def cache_video(self, file_path, video):
        if file_path in self.cached_video:
            return

        self.cached_video[file_path] = video
        # Remove the oldest cached video
        to_be_removed = None
        if self.cached_queue.qsize() >= self.max_queue:
            to_be_removed = self.cached_queue.get()

        if to_be_removed is not None:
            v = self.cached_video[to_be_removed]
            v.release()
            del self.cached_video[to_be_removed]
