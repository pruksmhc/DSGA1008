import os
import random

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
from tqdm import tqdm
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
import utils
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = 'data'
annotation_csv = 'data/annotation.csv'
unlabeled_scene_index = np.arange(106)
train_index = np.arange(106, 130)
val_index = np.arange(130, 134)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN


transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                   ])
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_index,
                                  transform=transform,
                                  extra_info=True
                                 )
labeled_valset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=val_index,
                                  transform=transform,
                                  extra_info=True
                                 )
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
num_classes  = 10
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
res = models.resnet50(pretrained = False, num_classes = 4)
res.load_state_dict(torch.load('self_sup/models/resnet50_best.pkl'))
model.backbone.body.layer1 = res.layer1
model.backbone.body.layer2 = res.layer2
model.backbone.body.layer3 = res.layer3
model.backbone.body.layer4 = res.layer4
model = model.cuda()


def rebatchify(batch_data):
    batch_data, targets, _, _ = batch_data
    batch_size = len(batch_data)
    return [torch.nn.functional.interpolate(torchvision.utils.make_grid(
        torch.stack([batch_data[i][0], batch_data[i][1], batch_data[i][2],
        torch.flip(batch_data[i][3],[1,2]),torch.flip(batch_data[i][4],[1,2]),
        torch.flip(batch_data[i][5],[1,2])]), nrow = 3, padding = 0).unsqueeze(0), (800, 800), mode="bilinear").squeeze(0)
        for i in range(batch_size)], targets

def target2boxes(targets):
    result = []
    for t in targets:
        boxes = torch.cat([(torch.FloatTensor([torch.min(bb[0,:]), torch.min(bb[1,:]), torch.max(bb[0,:]), torch.max(bb[1,:])]).unsqueeze(0) + 40)*10 for bb in t['bounding_box']])
        result.append({'boxes':boxes, 'labels': t['category']+1})
    return result

def boxes2targets(boxes):
    return torch.cat([torch.DoubleTensor([[bb[2], bb[2], bb[0], bb[0]], [bb[3],bb[1],bb[3],bb[1]]]).unsqueeze(0)/10 - 40 for bb in boxes])
img, targets = rebatchify(iter(trainloader).next())



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = rebatchify(batch_data)
        targets = target2boxes(targets)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area


def compute_ats_bounding_boxes(boxes1, boxes2):
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
    boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
    boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
    boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

    boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
    boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
    boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
    boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

    condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
    condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
    condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
    condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
    condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

    iou_matrix = torch.zeros(num_boxes1, num_boxes2)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            if condition_matrix[i][j]:
                print(boxes1[i])
                iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])

    iou_max = iou_matrix.max(dim=0)[0]

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    total_threat_score = 0
    total_weight = 0
    for threshold in iou_thresholds:
        tp = (iou_max > threshold).sum()
        threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
        total_threat_score += 1.0 / threshold * threat_score
        total_weight += 1.0 / threshold

    average_threat_score = total_threat_score / total_weight
    
    return average_threat_score


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    metric = []
    for batch_data in metric_logger.log_every(data_loader, 100, header):
        image, targets = rebatchify(batch_data)
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        
        
        outputs = [boxes2targets(t['boxes'].to(cpu_device)) for t in outputs]
        assert len(outputs) == len(targets)
        for i in range(len(targets)):
            metric.append(compute_ats_bounding_boxes(outputs[i], targets[i]['bounding_box'].to(cpu_device)))
        model_time = time.time() - model_time

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    return np.array(metric)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
num_of_epochs = 100
for epoch in range(num_of_epochs):
    train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq=10)
    torch.save(model.state_dict(),'bounding_box/model/fastrcnn_{}_{}.pkl'.format('Res50',epoch))


