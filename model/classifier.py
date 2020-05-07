import torch
import torch.nn as nn

class Classifier(nn.Modules):

    def __init__(self, device, number_classes, roi_head_classifier = None, cls_loc = None, score = None):
        self.device = device
        self.number_classes = number_classes
        
        if roi_head_classifier is None:
            self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                      nn.Linear(4096, 4096)]).to(device)
        else:
            self.roi_head_classifier = roi_head_classifier

        if cls_loc is None:
            self.cls_loc = nn.Linear(4096, self.number_classes * 4).to(device) 
            cls_loc.weight.data.normal_(0.01)
            cls_loc.bias.data.zero_()
        else:
            self.cls_loc = cls_loc

        if score is None:
            self.score = nn.Linear(4096, self.number_classes).to(device) 
        else:
            self.score = score

    def forward(self, output):
        k = output.view(output.size(0), -1)
        k = self.roi_head_classifier(k)
        roi_cls_loc = self.cls_loc(k)
        roi_cls_score = self.score(k)
        
        preds = roi_cls_loc.view(128, -1, 4)
        
        pred_class = torch.argmax(roi_cls_score, dim=1)
        preds_bbox = preds[torch.arange(0, 128).long(), pred_class]
        
        return roi_cls_loc, roi_cls_score, preds_bbox, pred_class
