import torch
import torch.nn as nn
import numpy as np

class AnchorGenerator(nn.Module):

    def __init__(self, img_size=800,  sub_sample = 16,
                 ratios = [0.5, 1, 2], anchor_scales = [8, 16, 32]):
        super(AnchorGenerator, self).__init__()
        self.img_size = img_size
        self.sub_sample = sub_sample
        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]
        

    def generate_anchor_boxes(self, bboxes, train):
        #number of anchors per center
        n_anchors = len(self.ratios) * len(self.anchor_scales)
        fe_size = (self.img_size // self.sub_sample)
        ctr_x = np.arange(self.sub_sample, (fe_size + 1) * self.sub_sample, self.sub_sample)
        ctr_y = np.arange(self.sub_sample, (fe_size + 1) * self.sub_sample, self.sub_sample)
        ctr = np.zeros((len(ctr_x) * len(ctr_y), 2))
        index = 0
        for x in range(len(ctr_x)):
            for y in range(len(ctr_y)):
                ctr[index, 1] = ctr_x[x] - (self.sub_sample//2)
                ctr[index, 0] = ctr_y[y] - (self.sub_sample//2)
                index += 1
        anchors = np.zeros((fe_size * fe_size * n_anchors, 4))
        index = 0
        for c in ctr:
            ctr_y, ctr_x = c
            for i in range(len(self.ratios)):
                for j in range(len(self.anchor_scales)):
                    h = self.sub_sample * self.anchor_scales[j] * np.sqrt(self.ratios[i])
                    w = self.sub_sample * self.anchor_scales[j] * np.sqrt(1. / self.ratios[i])
                    anchors[index, 0] = ctr_y - h / 2.
                    anchors[index, 1] = ctr_x - w / 2.
                    anchors[index, 2] = ctr_y + h / 2.
                    anchors[index, 3] = ctr_x + w / 2.
                    index += 1
        # Out: [22500, 4]
        index_inside = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= self.img_size) &
            (anchors[:, 3] <= self.img_size)
        )[0]
        valid_anchor_boxes = anchors[index_inside]
        label = np.empty((len(index_inside),), dtype=np.int32)
        label.fill(-1)
        if bboxes is None:
            bboxes = torch.Tensor([[200, 400, 600, 400], [200, 200, 600, 300]])
        ious = np.empty((len(valid_anchor_boxes), len(bboxes)), dtype=np.float32)
        ious.fill(0)
        for num1, i in enumerate(valid_anchor_boxes):
            ya1, xa1, ya2, xa2 = i
            anchor_area = (ya2 - ya1) * (xa2 - xa1)
            for num2, j in enumerate(bboxes):
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

        if train:
            # the highest iou for each gt_box and its corresponding anchor box
            gt_argmax_ious = ious.argmax(axis=0)
            gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
            # find the anchor_boxes which have this max_ious (gt_max_ious)
            gt_argmax_ious = np.where(ious == gt_max_ious)[0]
            label[gt_argmax_ious] = 1
        #the highest iou for each anchor box and its corresponding ground truth box
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(index_inside)), argmax_ious]

        pos_iou_threshold  = 0.7
        neg_iou_threshold = 0.3
        label[max_ious < neg_iou_threshold] = 0
        label[max_ious >= pos_iou_threshold] = 1

        max_iou_bbox = bboxes[argmax_ious]
        height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
        width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
        ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
        ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width
        base_height = max_iou_bbox[:,  2] - max_iou_bbox[:, 0]
        base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
        base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
        base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

        eps = np.finfo(height.dtype).eps
        height = np.maximum(height, eps)
        width = np.maximum(width, eps)
        dy = (base_ctr_y - ctr_y) / height
        dx = (base_ctr_x - ctr_x) / width
        dh = np.log(abs(base_height / height))
        dw = np.log(abs(base_width / width))
        anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

        anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
        anchor_labels.fill(-1)
        anchor_labels[index_inside] = label

        anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
        anchor_locations.fill(0)
        anchor_locations[index_inside, :] = anchor_locs

        return anchor_locations, anchor_labels, anchors

    def generate_anchor_boxes_test(self):
        raise NotImplementedError()

    def forward(self, bboxes = None, train=False):
        return self.generate_anchor_boxes_test(bboxes, train)
        
