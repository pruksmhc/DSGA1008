import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import torchvision.models as models
from anchor_generator import AnchorGenerator
from region_proposal_network import RegionProposalNetwork
from max_pooling_layer import MaxPoolingLayer

"""
Adapted from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
"""
device = "cuda"

class FasterRCNN(nn.Module):
    self.train = None
    self.device = None
    self.backbone = None
    self.img_size = None
    self.anchor_generator = None
    self.rpn_in_features = None
    self.region_proposal_network = None
    self.max_pooling_layer = None
    self.classifier = None

    def __init__(self, train = False, device = None, img_size = 800, 
                 rpn_in_features = 512, backbone = None, region_proposal_network = None,
                 max_pooling_layer = None, classifier = None):
        super(FasterRCNN, self).__init__()
        self.train = train
        if backbone is None:
            self.backbone = models.alexnet(pretrained=False, num_classes=4).to(device)
        
            # model.load_state_dict(torch.load(PATH))
            state_dict_file = "alexnet_5.pkl"
            if torch.cuda.is_available():
                self.alexnet.load_state_dict(torch.load(state_dict_file))
            else:
                self.alexnet.load_state_dict(torch.load(state_dict_file,
                                                        map_location=torch.device('cpu')))
        else:
            self.backbone = backbone

        self.img_size = img_size
        if anchor_generator is None:
            self.anchor_generator = AnchorGenerator(self.img_size, train = self.train)
        else:
            self.anchor_generator = anchor_generator

        self.rpn_in_features = rpn_in_features
        if region_proposal_network is None:
            self.region_proposal_network = RegionProposalNetwork(self.rpn_in_features, self.device, train = self.train)
        else:
            self.region_proposal_network = region_proposal_network

        if max_pooling_layer is None:
            self.max_pooling_layer = MaxPoolingLayer(sub_sample = self.sub_sample)
        else:
            self.max_pooling_layer = max_pooling_layer

    def compute_loss(self, samples):
        raise NotImplementedError()
                
    def forward(self, samples, bboxes = None, labels = None):
        # This is a not-transformed sample.
        img_features = self.alexnet.features(samples[0].to(device))
        img_features = torch.nn.functional.interpolate(img_features, size=(512,50,50))[0]
        # Generate anchor boxes
        anchor_locations, anchor_labels = self.achor_generator.forward(bboxes)
        
        sample_rois, gt_roi_locs, gt_roi_labels = self.region_proposal_network(img_features, bboxes, labels)

        output = self.max_pooling_layer(sample_rois, img_features)

        return classifier(output)
