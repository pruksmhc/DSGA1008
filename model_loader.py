"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import transform_back_bounding_boxes, transform_samples

# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # return torchvision.transforms.Compose([
    # 
    # 
    # ])
    
class ModelLoader():
    # Fill the information for your team
    team_name = ''
    team_member = []
    contact_email = '@nyu.edu'

    self.bbox_model = None
    self.lane_model = None

    def __init__(model_file):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.bbox_model = FasterRCNN()
        #state_dict_file_lane = 'state_dict_faster_rcnn.pkl'
        #if torch.cuda.is_available():
        #    self.model.load_state_dict(torch.load(state_dict_file_bbox)))
        #else:
        #    self.model.load_state_dict(state_dict_file,
        #                               map_location=torch.device('cpu')))
        self.bbox_model = self.bbox_model.to(device)

        self.lane_model = 
        #state_dict_file_lane = 'state_dict_lane.pkl'
        #if torch.cuda.is_available():
        #    self.model.load_state_dict(torch.load(state_dict_file_lane)))
        #else:
        #    self.model.load_state_dict(state_dict_file_lane,
        #                               map_location=torch.device('cpu')))# You should 
        self.lane_model = self.lane_model.to(device)
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 

    def get_bounding_boxes(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        bboxes = self.bbox_model.forward(transform_samples(samples).to(device))
        return transform_back_bounding_boxes(bboxes)

    def get_binary_road_map(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 80] 
        return self.lane_model.forward(transform_samples(samples).to(device))
