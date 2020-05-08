import os
import random
import math
import sys
import numpy as np
import pandas as pd
from helper import *
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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '/gpfs/scratch/wz727/DL/data'
annotation_csv = '/gpfs/scratch/wz727/DL/data/annotation.csv'
unlabeled_scene_index = np.arange(106)
train_index = np.arange(106, 130)
val_index = np.arange(130, 134)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ObjectDetector:

	def __init__(self, num_classes, pretrained=False):

		self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
		self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, num_classes)


	def boxes2targets(self, boxes):
	    if boxes.size(0) > 0:
	        return torch.cat([torch.DoubleTensor([[bb[2], bb[2], bb[0], bb[0]], [bb[3],bb[1],bb[3],bb[1]]]).unsqueeze(0)/10 - 40 for bb in boxes])
	    else:
	        return torch.zeros(1, 2, 4)