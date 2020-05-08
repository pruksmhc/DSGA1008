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
from roadmap import * 
from object_detection import * 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    return transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                   ])

class ModelLoader():
    # Fill the information for your team
    team_name = 'Radio Station WXYB'
    team_member = ["Xiaoyi Zhang", "Yada Pruksachatkun", 
                   "Weicheng Zhu", "Brian Kelly"]
    contact_email = 'bak438@nyu.edu'

    def __init__(self, model_file="models"):
        
        # self.bbox_model = FasterRCNNBoundingBox()
        # state_dict_file_lane = 'state_dict_faster_rcnn.pkl'
        # if torch.cuda.is_available():
        #    self.model.load_state_dict(torch.load(state_dict_file_bbox))
        # else:
        #    self.model.load_state_dict(state_dict_file,
        #                               map_location=torch.device('cpu'))
        # self.bbox_model = self.bbox_model.to(device)

        self.detector = ObjectDetector(num_classes = 10, pretrained=False)
        self.bbox_model = self.detector.model

        state_dict_file_bbox = model_file + '/state_dict_bounding_box.pkl'
        if torch.cuda.is_available():
           self.bbox_model.load_state_dict(torch.load(state_dict_file_bbox))
        else:
           self.bbox_model.load_state_dict(torch.load(state_dict_file_bbox,
                                      map_location=torch.device('cpu')))
        self.bbox_model = self.bbox_model.to(device)


        self.lane_model = Unet(backbone_name='resnet101', pretrained=False, encoder_freeze=False, classes=2)
        state_dict_file_lane = model_file + '/state_dict_road_map.pkl'
        if torch.cuda.is_available():
           self.lane_model.load_state_dict(torch.load(state_dict_file_lane))
        else:
           self.lane_model.load_state_dict(torch.load(state_dict_file_lane,
                                      map_location=torch.device('cpu')))
        self.lane_model = self.lane_model.to(device)
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        self.bbox_model.eval()
        outputs = self.bbox_model.forward(transform_samples(samples).to(device))
        return [self.detector.boxes2targets(t['boxes']) for t in outputs]


    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 80] 
        self.lane_model.eval()
        return torch.max(self.lane_model.forward(transform_samples(samples).to(device)).data, 1)[1]

        