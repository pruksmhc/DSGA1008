import torch
import torchvision

def transform_bounding_box(self, bboxes):
    new_bboxes = []
    for bbox in bboxes:
        x1 = (bbox[1][0] + bbox[1][2])/2 + 400
        x2 = (bbox[1][1] + bbox[1][3])/2 + 400
        y1 = (bbox[0][0] + bbox[0][2])/2 + 400
        y2 = (bbox[0][1] + bbox[0][3])/2 + 400
        #anchors[index, 0] = ctr_y - h / 2.
        #anchors[index, 1] = ctr_x - w / 2.
        #anchors[index, 2] = ctr_y + h / 2.
        #anchors[index, 3] = ctr_x + w / 2.
        new_bboxes.append([min(y1, y2), min(x1, x2), max(y1, y2), max(x1,x2)])
    return torch.Tensor(new_bboxes)

def transform_sample(samples):
    return torch.stack([torchvision.utils.make_grid(
        torch.stack([batch_data[i][0], batch_data[i][1], batch_data[i][2],
                     torch.flip(batch_data[i][3],[1,2]),torch.flip(batch_data[i][4],[1,2]),
                     torch.flip(batch_data[i][5],[1,2])]), nrow = 3, padding = 0)
                        for i in range(batch_size)])

def transform_back_bounding_boxes(bboxes):
    output = torch.zeros([len(bboxes),2,4], dtype=torch.double)
    for i in range(len(bboxes)):
        #x1 = (bbox[1][0] + bbox[1][2])/2 + 400
        #x2 = (bbox[1][1] + bbox[1][3])/2 + 400
        #y1 = (bbox[0][0] + bbox[0][2])/2 + 400
        #y2 = (bbox[0][1] + bbox[0][3])/2 + 400
        output[i, 1, 0] = output[i, 1, 2] = bboxes[i, 1] - 400
        output[i, 1, 1] = output[i, 1, 3] = bboxes[i, 3] - 400
        output[i, 0, 0] = output[i, 0, 2] = bboxes[i, 0] - 400
        output[i, 0, 1] = output[i, 0, 3] = bboxes[i, 2] - 400
    return output

def rebatchify(batch_data):
  batch_data, targets, batch_road, extra = batch_data
  batch_size = len(batch_data)
  batch_data = torch.stack([torchvision.utils.make_grid(
        torch.stack([batch_data[i][0], batch_data[i][1], batch_data[i][2],
        torch.flip(batch_data[i][3],[1,2]),torch.flip(batch_data[i][4],[1,2]),
        torch.flip(batch_data[i][5],[1,2])]), nrow = 3, padding = 0)
        for i in range(batch_size)])
  for i in range(batch_size):
      targets = targets[i]['bounding_box'] = transform_bounding_box(targets[i]['bounding_box'])
  batch_road = torch.stack(batch_road).long()
  extra = extra
  return  batch_data, targets, batch_road, extra
