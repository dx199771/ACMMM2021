
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

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.data

    return hook


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.avgpool.register_forward_hook(get_activation('avgpool'))

    def extract(self, img):
        self.model.eval()
        # compute output
        with torch.no_grad():
            output = self.model(img)

        feature = torch.flatten(activation['avgpool'], 1)
        feature = feature / torch.norm(feature[0])
        return feature
