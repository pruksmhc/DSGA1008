import torch
import torch.nn as nn
import numpy as np

class MaxPoolingLayer(nn.Modules):

    def __init__(self, size = 7, sub_sample = 16, adaptive_max_pool = None):
        super(MaxPoolingLayer, self).__init__()
        self.size = size
        self.sub_sample = sub_sample
        if adaptive_max_pool is None:
            self.adaptive_max_pool = nn.AdaptiveMaxPool2d((self.size, self.size))
        else:
            self.adaptive_max_pool = adaptive_max_pool

    def forward(self, sample_rois, img_features):
        size = (7, 7)
        adaptive_max_pool = nn.AdaptiveMaxPool2d((size[0], size[1]))
        rois = torch.from_numpy(sample_rois).float()
        roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
        roi_indices = torch.from_numpy(roi_indices).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        
        output = []
        rois = indices_and_rois.data.float()
        rois[:, 1:].mul_(1/self.sub_sample) # Subsampling ratio
        rois = rois.long()
        num_rois = rois.size(0)
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
        im = img_features.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(adaptive_max_pool(im))
        output = torch.cat(output, 0)
        return output
