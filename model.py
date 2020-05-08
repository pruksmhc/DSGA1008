import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import torchvision.models as models
import anchor_generator
from anchor_generator import AnchorGenerator
import region_proposal_network
from region_proposal_network import RegionProposalNetwork
import max_pooling_layer
from max_pooling_layer import MaxPoolingLayer
import classifier
from classifier import Classifier

"""
Adapted from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
"""
device = "cuda"

class FasterRCNNBoundingBox(nn.Module):

    def __init__(self, device = None, img_size = 800,
                 rpn_in_features = 512, backbone = None, region_proposal_network = None,
                 max_pooling_layer = None, sub_sample=16):
        super(FasterRCNNBoundingBox, self).__init__()
        self.device = device
        self.sub_sample = sub_sample
        self.classifier = Classifier(device, number_classes=9)
        if backbone is None:
            self.backbone = models.alexnet(pretrained=False, num_classes=4).to(device)
            
            state_dict_file = "alexnet_5.pkl"
            if torch.cuda.is_available():
                self.backbone.load_state_dict(torch.load(state_dict_file))
            else:
                self.backbone.load_state_dict(torch.load(state_dict_file,
                                                        map_location=torch.device('cpu')))
            
        else:
            self.backbone = backbone

        self.img_size = img_size
        self.anchor_generator = AnchorGenerator(self.img_size )

        self.rpn_in_features = rpn_in_features
        self.region_proposal_network = RegionProposalNetwork(self.rpn_in_features, self.device)

        if max_pooling_layer is None:
            self.max_pooling_layer = MaxPoolingLayer(sub_sample = self.sub_sample)
        else:
            self.max_pooling_layer = max_pooling_layer

    def compute_loss(self, samples):
        raise NotImplementedError()
                
    def forward(self, samples, bboxes = None, labels = None, train=False):
        """
        Args:
            samples: This is the transformed images.
            bboxes: This is the target bounding boxes.
            labels: This is the target categories of each bounding box.
            train: bool

        Returns:
            Output locations, output scores, and a bunch of things you need to calculate the loss.
        """
        import pdb#; pdb.set_trace()
        img_features = self.backbone.features(samples.to(device))
        img_features = torch.nn.functional.interpolate(img_features[None], size=(512,50,50))[0]

        
        # Generate anchor boxes
        anchor_locations, anchor_labels, anchors = self.anchor_generator.forward(bboxes)
        
        sample_rois, gt_roi_locs, gt_roi_labels = self.region_proposal_network.forward(img_features, bboxes, anchors, labels)

        output = self.max_pooling_layer.forward(sample_rois, img_features)
        if train:
            # Get the logits
            output_loc, output_score, _, _ = self.classifier.forward(output)
        else:
            # Get the predictions
            _, _, output_loc, output_score = self.classifier(output)
        return output_loc, output_score, anchor_locations, anchor_labels, gt_roi_locs, gt_roi_labels
