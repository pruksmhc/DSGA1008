mport torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import torchvision.models as models

"""
Adapted from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
"""
device = "cuda"

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.alexnet = models.alexnet(pretrained=False, num_classes=4).to(device)

        # model.load_state_dict(torch.load(PATH))
        state_dict_file = "model/alexnet_5.pkl"
        if torch.cuda.is_available():
            self.alexnet.load_state_dict(torch.load(state_dict_file))
        else:
            self.alexnet.load_state_dict(torch.load(state_dict_file,
                                               map_location=torch.device('cpu')))

    def update_bounding_box(self, bboxes):
      new_bboxes = []
      for bbox in bboxes:
        x1 = (bbox[1][0] + bbox[1][2])/2
        x2 = (bbox[1][1] + bbox[1][3])/2
        y1 = (bbox[0][0] + bbox[0][2])/2
        y2 = (bbox[0][0] + bbox[0][3])/2
        new_bboxes.append([y1, x1, y2, x2])
      return torch.Tensor(new_bboxes)

    def compute_loss(self, samples):

    def forward(self, samples):
        # This is a not-transformed sample.
        img_features = self.alexnet.features(samples[0].to(device))
        img_features = torch.nn.functional.interpolate(img_features, size=(512,50,50))[0]
        # Generate anchor boxes
        import numpy as np
        ratios = [0.5, 1, 2]
        anchor_scales = [8, 16, 32]

        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
        fe_size = (800 // 16)
        ctr_x = np.arange(16, (fe_size + 1) * 16, 16)
        ctr_y = np.arange(16, (fe_size + 1) * 16, 16)
        ctr = np.zeros((len(ctr_x) * len(ctr_y), 2))
        index = 0
        for x in range(len(ctr_x)):
            for y in range(len(ctr_y)):
                ctr[index, 1] = ctr_x[x] - 8
                ctr[index, 0] = ctr_y[y] - 8
                index += 1
        anchors = np.zeros((fe_size * fe_size * 9, 4))
        index = 0
        sub_sample = 16
        for c in ctr:
            ctr_y, ctr_x = c
            for i in range(len(ratios)):
                for j in range(len(anchor_scales)):
                    h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                    w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
                    anchors[index, 0] = ctr_y - h / 2.
                    anchors[index, 1] = ctr_x - w / 2.
                    anchors[index, 2] = ctr_y + h / 2.
                    anchors[index, 3] = ctr_x + w / 2.
                    index += 1
        # Out: [22500, 4]
        index_inside = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= 800) &
            (anchors[:, 3] <= 800)
        )[0]
        valid_anchor_boxes = anchors[index_inside]
        label = np.empty((len(index_inside),), dtype=np.int32)
        label.fill(-1)
        ious = np.empty((len(valid_anchor_boxes), len(bbox)), dtype=np.float32)
        ious.fill(0)
        print(bbox)
        for num1, i in enumerate(valid_anchor_boxes):
            ya1, xa1, ya2, xa2 = i
            anchor_area = (ya2 - ya1) * (xa2 - xa1)
            for num2, j in enumerate(bbox):
                yb1, xb1, yb2, xb2 = j
                box_area = (yb2 - yb1) * (xb2 - xb1)
                inter_x1 = max([xb1, xa1])
                inter_y1 = max([yb1, ya1])
                inter_x2 = min([xb2, xa2])
                inter_y2 = min([yb2, ya2])
                if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                    iter_area = (inter_y2 - inter_y1) * \
                                (inter_x2 - inter_x1)
                    iou = iter_area / \
                          (anchor_area + box_area - iter_area)
                else:
                    iou = 0.
                ious[num1, num2] = iou
                argmax_ious = ious.argmax(axis=1)
                print(argmax_ious.shape)
                print(argmax_ious)
                max_ious = ious[np.arange(len(index_inside)), argmax_ious]