import torch
import torch.nn as nn
import numpy as np

class RegionProposalNetwork(nn.Module):

    def __init__(self, in_channels, device, mid_channels = 512, n_anchor = 9, image_size = 800,
                 nms_thresh = 0.7, min_size = 16, n_train_pre_nms = 12000,
                 n_train_post_nms = 2000, n_test_pre_nms = 6000, n_test_post_nms = 300,
                 conv1 = None, reg_layer = None, cls_layer = None,
                 n_sample = 128, pos_ratio = 0.25, pos_iou_thresh = 0.5, neg_iou_thresh_hi = 0.5,
                 neg_iou_thresh_lo = 0.0, pos_roi_per_image = 32):
        super(RegionProposalNetwork, self).__init__()
        self.in_channels = in_channels
        self.device = device
        self.mid_channels = mid_channels
        self.n_anchor = n_anchor
        self.image_size = image_size
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.pos_roi_per_image = pos_roi_per_image

        if conv1 is None:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(device)
            # conv sliding layer
            conv1.weight.data.normal_(0, 0.01)
            conv1.bias.data.zero_()
        else:
            self.conv1 = conv1
        if reg_layer  is None:
            self.reg_layer = nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0).to(device)
            # Regression layer
            reg_layer.weight.data.normal_(0, 0.01)
            reg_layer.bias.data.zero_()
        else:
            self.reg_layer = reg_layer
        if cls_layer is None:
            self.cls_layer = nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0).to(device)
            # classification layer
            cls_layer.weight.data.normal_(0, 0.01)
            cls_layer.bias.data.zero_()
        else:
            self.cls_layer = cls_layer

        def generate_proposal_rois(self, features, anchors, train):
            if train:
                pre_nms = self.n_train_pre_nms
                post_nms = self.n_train_post_nms
            else:
                pre_nms = self.n_test_pre_nms
                post_nms = self.n_test_post_nms

            x = conv1(features.to(device)) # out_map is obtained in section 1
            pred_anchor_locs = reg_layer(x)
            pred_cls_scores = cls_layer(x)

            pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
            #Out: torch.Size([1, 22500, 4])
            pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
            #Out torch.Size([1, 50, 50, 18])
            objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
            #Out torch.Size([1, 22500])
            pred_cls_scores  = pred_cls_scores.view(1, -1, 2)

            anc_height = anchors[:, 2] - anchors[:, 0]
            anc_width = anchors[:, 3] - anchors[:, 1]
            anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
            anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

            pred_anchor_locs_numpy = pred_anchor_locs[0].cpu().data.numpy()
            objectness_score_numpy = objectness_score[0].cpu().data.numpy()
            
            dy = pred_anchor_locs_numpy[:, 0::4]
            dx = pred_anchor_locs_numpy[:, 1::4]
            dh = pred_anchor_locs_numpy[:, 2::4]
            dw = pred_anchor_locs_numpy[:, 3::4]
            ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
            ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
            h = np.exp(dh) * anc_height[:, np.newaxis]
            w = np.exp(dw) * anc_width[:, np.newaxis]
            roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
            roi[:, 0::4] = ctr_y - 0.5 * h
            roi[:, 1::4] = ctr_x - 0.5 * w
            roi[:, 2::4] = ctr_y + 0.5 * h
            roi[:, 3::4] = ctr_x + 0.5 * w

            roi[:, slice(0, 4, 2)] = np.clip(
                roi[:, slice(0, 4, 2)], 0, self.img_size)
            roi[:, slice(1, 4, 2)] = np.clip(
                roi[:, slice(1, 4, 2)], 0, self.img_size)

            hs = roi[:, 2] - roi[:, 0]
            ws = roi[:, 3] - roi[:, 1]
            keep = np.where((hs >= min_size) & (ws >= min_size))[0]
            roi = roi[keep, :]
            score = objectness_score_numpy[keep]

            order = score.ravel().argsort()[::-1]
            order = order[:pre_nms]
            roi = roi[order, :]

            y1 = roi[:, 0]
            x1 = roi[:, 1]
            y2 = roi[:, 2]
            x2 = roi[:, 3]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = score[:n_train_pre_nms].argsort()[::-1]
            keep = []
            while order.size > 0:
                #print(order.size)
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (area[i] + area[order[1:]] - inter)
                inds = np.where(ovr <= nms_thresh)[0]
                order = order[inds + 1]

                keep = keep[:post_nms] # while training/testing , use accordingly
                roi = roi[keep] # the final region proposals

            return roi

        def generate_sample_rois(self, proposal_rois, labels, train):
            bboxes = torch.Tensor([[200, 400, 600, 400], [200, 200, 600, 300]])
            ious = np.empty((len(proposal_rois), len(bboxes)), dtype=np.float32)
            ious.fill(0)
            for num1, i in enumerate(proposal_rois):
                ya1, xa1, ya2, xa2 = i  
                anchor_area = (ya2 - ya1) * (xa2 - xa1)
                for num2, j in enumerate(bboxes):
                    yb1, xb1, yb2, xb2 = j
                    box_area = (yb2- yb1) * (xb2 - xb1)
                    inter_x1 = max([xb1, xa1])
                    inter_y1 = max([yb1, ya1])
                    inter_x2 = min([xb2, xa2])
                    inter_y2 = min([yb2, ya2])
                    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                        iter_area = (inter_y2 - inter_y1) * \
                                    (inter_x2 - inter_x1)
                        iou = iter_area / (anchor_area+ \
                                           box_area - iter_area)            
                    else:
                        iou = 0.
                    ious[num1, num2] = iou

            max_iou = ious.max(axis=1)
            pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
            pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
            if pos_index.size > 0:
                pos_index = np.random.choice(
                    pos_index, size=pos_roi_per_this_image, replace=False)

            neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                                 (max_iou >= self.neg_iou_thresh_lo))[0]
            neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
            neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                             neg_index.size))
            if  neg_index.size > 0 :
                neg_index = np.random.choice(
                    neg_index, size=neg_roi_per_this_image, replace=False)
                    
            keep_index = np.append(pos_index, neg_index)
            sample_roi = proposal_rois[keep_index]
            gt_roi_labels = None
            gt_assignment = None
            if train:
                gt_assignment = ious.argmax(axis=1)
                gt_roi_label = labels[gt_assignment]
                gt_roi_labels = gt_roi_label[keep_index]
                gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0

            return sample_roi, gt_assignment, gt_roi_labels

        def forward(self, features, bboxes = None, anchors=None, labels = None, train=False):
            if train and bboxes is None:
                raise ValueError("train is true, but no bbox is passed!!")
            if train and labels is None:
                raise ValueError("train is true, but no label is passed!!")

            proposal_rois = self.generate_proposal_rois(features, anchors, train)
            return self.generate_sample_rois(proposal_rois, labels)
