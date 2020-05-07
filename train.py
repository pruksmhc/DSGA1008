from model import model
from model.model import FasterRCNNBoundingBox
import torch
##### import os
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = 'data'
annotation_csv = 'data/annotation.csv'
unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = FasterRCNNBoundingBox(device=device)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = torchvision.transforms.ToTensor()
transforms_list_augmentation = transforms.Compose([transforms.Resize(256),
                                                    transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.ToTensor()])
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=True
                                 )

trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)
samples, targets, road_image, extra = next(iter(trainloader))
model(samples, train=False)



