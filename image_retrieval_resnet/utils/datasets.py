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
from PIL import Image, ExifTags, ImageOps
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


def create_dataloader(path, batch_size, cache=True, transform=None, sampler=None, use_instance_id=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    dataset = LoadImagesAndLabels(path, transform=transform,
                                  cache_images=cache, use_instance_id=use_instance_id)

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


class LoadImagesAndLabels(Dataset):  # for training/testing
    is_six_column = False

    def __init__(self, path, transform=None, cache_images=False, use_instance_id=False):
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

        self.use_instance_id = use_instance_id
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
        if len(self.data) > 0:
            if len(self.data[0]['label']) == 6:
                self.is_six_column = True

        self.labels = []
        if self.use_instance_id:
            for i in range(0, len(self.data)):
                self.labels.append(int(self.data[i]['label'][1]))
        else:
            for i in range(0, len(self.data)):
                self.labels.append(int(self.data[i]['label'][0]))

        self.class_to_idx = cache['class_to_idx']

        self.n = len(self.data)  # number of samples
        self.cache_images = cache_images
        self.images = [None] * self.n
        self.transform = transform

    def cache_data(self, path='data.cache'):
        # Cache dataset labels, check images and read shapes
        x = {"data": [], "class_to_idx": {}}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Loading labels', total=len(self.img_files))
        now_sn = 0
        for (img, label) in pbar:
            l = []
            image = Image.open(img)
            image.verify()  # PIL verify
            # _ = io.imread(img)  # skimage verify (from skimage import io)
            shape = exif_size(image)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
            if os.path.isfile(label):
                with open(label, 'r') as f:
                    origin_l = [x.split() for x in f.read().splitlines()]
                    if self.use_instance_id:
                        if len(origin_l[0]) != 6:
                            raise ValueError("number of column when use instance id should be 6!")

                    if len(origin_l[0]) == 6 and self.use_instance_id:
                        for line in origin_l:
                            if line[1] not in x['class_to_idx']:
                                x['class_to_idx'][line[1]] = now_sn
                                now_sn += 1

                            # Substitute with index
                            line[1] = x['class_to_idx'][line[1]]
                    else:
                        for line in origin_l:
                            if line[0] not in x['class_to_idx']:
                                x['class_to_idx'][line[0]] = int(line[0])

                            # Substitute with index
                            line[0] = x['class_to_idx'][line[0]]

                    l = np.array(origin_l, dtype=np.float32)  # labels

            if len(l) == 0:
                if self.use_instance_id:
                    l = np.zeros((0, 6), dtype=np.float32)
                else:
                    l = np.zeros((0, 5), dtype=np.float32)

            for one_l in l:
                x["data"].append({'img': img, 'shape': shape, 'label': one_l})

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
        if self.cache_images and self.images[index] is not None:
            img = self.images[index]
        else:
            try:
                img, b = self.load_image(index)
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
        img, _ = self.load_image(index)

        return img

    def get_converted_item(self, index):
        item = self.data[index]
        cls = item["label"][1] if self.use_instance_id else item["label"][0]
        bbox = self.get_converted_bbox(index)
        return {"img": item['img'], 'class': cls, 'bbox': bbox}

    def get_converted_bbox(self, index):
        item = self.data[index]

        h, w = item['shape']

        if self.is_six_column:
            b = item['label'][2:] * [w, h, w, h]  # box
        else:
            b = item['label'][1:] * [w, h, w, h]  # box

        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

        return np.array([b[0], b[1], b[2], b[3]]).tolist()

    def load_image(self, index):
        # loads 1 image from dataset, returns clipped img
        item = self.data[index]

        p = Path(item['img'])
        img = Image.open(str(p)).convert('RGB')
        img = np.array(img)
        h, w = item['shape']

        if self.is_six_column:
            b = item['label'][2:] * [w, h, w, h]  # box
        else:
            b = item['label'][1:] * [w, h, w, h]  # box

        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

        orginal_xywh = np.array(b)
        '''
        # original_size = np.array(b[2:])
        b[2:] = b[2:].max()  # rectangle to square
        b[2:] = b[2:] * 1.1 + 20  # pad
        # enlarged = b[2:] - original_size  # enlarged after changing to square and padding
        # b[:2] = b[:2] - enlarged / 2  # Move the left top point to make sure the object is in the middle

        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

        original_b = np.array(b)
        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)

        padding = np.zeros(4)
        padding[:2] = b[:2] - original_b[:2]
        padding[2:] = original_b[2:] - b[2:]
        padding = padding.astype(np.int)

        return_img = None
        if np.all((padding == 0)):
            return_img = Image.fromarray(img[b[1]:b[3], b[0]:b[2]])
        else:
            return_img = ImageOps.expand(Image.fromarray(img[b[1]:b[3], b[0]:b[2]]), (padding[0], padding[1], padding[2], padding[3]),
                                         fill='black')  # add border
        '''
        return_img = Image.fromarray(img[b[1]:b[3], b[0]:b[2]])

        original_xyxy = xywh2xyxy(orginal_xywh.reshape(-1, 4)).ravel().astype(np.int)

        return return_img, original_xyxy
