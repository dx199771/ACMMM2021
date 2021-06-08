import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.general import torch_distributed_zero_first, xywh2xyxy

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = (img.size[1], img.size[0])  # (0 is width, 1 is height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception as e:
        pass

    return s


def create_dataloader(path, batch_size, cache=True, transform=None, sampler=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    print(path)
    dataset = LoadImagesAndLabels(path, batch_size, transform=transform, cache_images=cache)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 1, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=nw,
                                             pin_memory=True)
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, batch_size=16, transform=None, cache_images=False):
        try:
            f = []  # image files
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
            self.img_files = sorted(
                [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))
        image_number = len(self.img_files)
        assert image_number > 0, 'No images found in %s' % (path)

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.img_files]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_data(cache_path)  # re-cache
        else:
            cache = self.cache_data(cache_path)  # cache

        # data stores all the parameters of all the samples
        self.data = cache['data']

        self.n = len(self.data)  # number of samples
        self.cache_images = cache_images
        self.imgs = [None] * self.n
        self.transform = transform

    def cache_data(self, path='data.cache'):
        # Cache dataset labels, check images and read shapes
        x = {"data": []}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Loading labels', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)

                for one_l in l:
                    x["data"].append({'img': img, 'shape': shape, 'label': one_l})

            except Exception as e:
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
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
        if self.cache_images and self.imgs[index] is not None:
            img = self.imgs[index]
        else:
            try:
                img, b = self.load_image(index)
                img = Image.fromarray(img)
            except Exception as e:
                print(item)
                print(e)

            if self.transform is not None:
                img = self.transform(img)

            if self.cache_images:
                self.imgs[index] = img

        return img, int(item['label'][0])

    def get_item(self, index):
        return self.data[index]

    def get_image(self, index):
        img, _ = self.load_image(index)

        return Image.fromarray(img)

    def get_converted_bbox(self, index):
        item = self.data[index]

        h, w = item['shape']

        b = item['label'][1:] * [w, h, w, h]  # box
        b[2:] = b[2:].max()  # rectangle to square
        b[2:] = b[2:] * 1.1 + 20  # pad
        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)

        return np.array([b[1], b[0], b[3], b[2]]).tolist()

    def load_image(self, index):
        # loads 1 image from dataset, returns clipped img
        item = self.data[index]

        p = Path(item['img'])
        img = Image.open(str(p)).convert('RGB')
        img = np.array(img)
        h, w = item['shape']

        b = item['label'][1:] * [w, h, w, h]  # box
        b[2:] = b[2:].max()  # rectangle to square
        b[2:] = b[2:] * 1.1 + 20  # pad
        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)

        return img[b[1]:b[3], b[0]:b[2]], b
