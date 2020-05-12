import os
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

transform = torchvision.transforms.ToTensor()
transforms_list_augmentation = transforms.Compose([transforms.Resize(256),
                                                    transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.ToTensor()])
unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=labeled_scene_index, first_dim='sample', transform=transforms_list_augmentation)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=3, shuffle=True, num_workers=2)

def rebatchify(batch):

	# rotation
    batch_size, num_of_angles, C, W, H = batch.size()
    batch = batch.contiguous().view(batch_size * num_of_angles, C, W, H)
    rotation_labels = torch.LongTensor([0, 1, 2, 3] * batch_size * num_of_angles)
    dataX_90 = torch.flip(torch.transpose(batch,2,3),[2])
    dataX_180 = torch.flip(torch.flip(batch,[2]),[3])
    dataX_270 = torch.transpose(torch.flip(batch,[2]),2,3)
    data = torch.stack([batch, dataX_90, dataX_180, dataX_270], dim=1)
    data = data.contiguous().view(batch_size * num_of_angles * 4, C, W, H)
    return data, rotation_labels

    # permutation
    # batch_size, num_of_angles, C, W, H = batch.size()
    # batch = batch.contiguous().view(batch_size * num_of_angles, C, W, H)
    # rotation_labels = torch.LongTensor([0, 1, 2, 3, 4, 5] * batch_size)
    # data = data.contiguous().view(batch_size * 6, C, W, H)
    # return data, permutation_labels


res50 = models.resnet50(pretrained=False, num_classes=4).to('cuda')

def train(data, model, optim, criterion, max_clip_norm=5):
    model.train()
    optim.zero_grad()
    data, labels = rebatchify(data)
    data, labels = data.to('cuda'), labels.to('cuda')
    logits = model(data)
    loss = criterion(logits, labels)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    loss.backward()
    optim.step()
    return loss.item()

lr = 3e-4
epoch = 50
optimizer = optim.Adam([p for p in res50.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
for epoch in range(epoch):
    total_loss = 0
    t = tqdm(iter(trainloader), leave=True, total=len(trainloader))
    for idx, batch_data in enumerate(t):
        batch_size = batch_data.size()[0]
        loss = train(batch_data, res50, optimizer, criterion, 5)
        total_loss += loss * batch_size
        if idx % 50 == 0:
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch + 1, total_loss))
            t.refresh()
    torch.save(res50.state_dict(),'self_sup/models/res50{}.pkl'.format(epoch + 1))
    