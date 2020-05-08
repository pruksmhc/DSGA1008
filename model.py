import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import anchor_generator
from anchor_generator import AnchorGenerator
import region_proposal_network
from region_proposal_network import RegionProposalNetwork
import max_pooling_layer
from max_pooling_layer import MaxPoolingLayer
import classifier
from classifier import Classifier
from utils import regression_loss

"""
Adapted from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
"""

class FasterRCNNBoundingBox(nn.Module):

    def __init__(self, device = 'cuda', img_size = 800, rpn_in_features = 512, sub_sample=16,
                 number_classes = 10, roi_lambda = 10, train = False,
                 backbone = None, anchor_generator = None, region_proposal_network = None, 
                 max_pooling_layer = None, classifier = None ):
        super(FasterRCNNBoundingBox, self).__init__()
        self.device = device
        self.sub_sample = sub_sample
        self.img_size = img_size
        self.rpn_in_features = rpn_in_features
        self.sub_sample = sub_sample
        self.number_classes = number_classes
        self.roi_lambda = roi_lambda
        self.train = train

        if backbone is None:
            self.backbone = models.alexnet(pretrained=False, num_classes=4).to(self.device)
            
            state_dict_file = "alexnet_5.pkl"
            if torch.cuda.is_available():
                self.backbone.load_state_dict(torch.load(state_dict_file))
            else:
                self.backbone.load_state_dict(torch.load(state_dict_file,
                                                        map_location=torch.device('cpu')))
            
        else:
            self.backbone = backbone

        self.img_size = img_size
        if anchor_generator is None:
            self.anchor_generator = AnchorGenerator(self.img_size )
        else:
            self.anchor_generator = anchor_generator

        self.rpn_in_features = rpn_in_features
        if region_proposal_network is None:
            self.region_proposal_network = RegionProposalNetwork(self.rpn_in_features, self.device)
        else:
            self.region_proposal_network = region_proposal_network

        if max_pooling_layer is None:
            self.max_pooling_layer = MaxPoolingLayer(sub_sample = self.sub_sample)
        else:
            self.max_pooling_layer = max_pooling_layer

        if classifier is None:
            self.classifier = Classifier(self.device, number_classes = self.number_classes)
        else:
            self.classifier = classifier

    def compute_roi_loss(roi_cls_loc, roi_cls_score, gt_roi_locs, gt_roi_labels):
        #classification loss
        gt_roi_loc = torch.from_numpy(gt_roi_locs).to(self.device)
        gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long().to(self.device)
        roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
        
        #regression loss
        n_sample = roi_cls_loc.shape[0]
        roi_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
        roi_loc_loss = regression_loss(roi_loc, gt_roi_loc)

        #regularize loss
        roi_loss = roi_cls_loss + (self.roi_lambda * roi_loc_loss)
        self.roi_loss = roi_loss

    def compute_total_loss(self, output_loc, output_score, gt_roi_locs, gt_roi_labels,
                           anchor_locations, anchor_labels):
        rpn_loss = self.region_proposal_network(anchor_locations, anchor_labels)
        roi_loss = compute_roi_loss(output_loc, output_score, gt_roi_locs, gt_roi_labels)

        self.total_loss = rpn_loss + roi_loss
        
    def get_roi_loss(self):
        return self.roi_loss

    def get_rpn_loss(self):
        return region_roposal_network.get_rpn_loss()

    def get_total_loss(self):
        return self.get_total_loss
    
    def train_mode(self):
        self.train = True

    def test_mode(self):
        self.train = False

    def forward(self, samples, bboxes = None, labels = None):
        """
        Args:
            samples: This is the transformed images.
            bboxes: This is the target bounding boxes.
            labels: This is the target categories of each bounding box.
            train: bool

        Returns:
            Output locations, output scores, and a bunch of things you need to calculate the loss.
        """

        img_features = self.backbone.features(samples.to(self.device))
        img_features = torch.nn.functional.interpolate(img_features[None], size=(512,50,50))[0]

        
        # Generate anchor boxes
        anchor_locations, anchor_labels, anchors = self.anchor_generator.forward(bboxes)
        
        sample_rois, gt_roi_locs, gt_roi_labels = self.region_proposal_network.forward(img_features, bboxes, anchors, labels)

        output = self.max_pooling_layer.forward(sample_rois, img_features)
        if self.train:
            # Get the logits
            output_loc, output_score, _, _ = self.classifier.forward(output)
            self.loss = compute_total_loss(output_loc, output_score, gt_roi_locs, gt_roi_labels, 
                                           anchor_locations, anchor_labels)
        else:
            # Get the predictions
            _, _, output_loc, output_score = self.classifier.forward(output)
        return output_loc, output_score, anchor_locations, anchor_labels, gt_roi_locs, gt_roi_labels
