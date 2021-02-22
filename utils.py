import cv2
import numpy as np
import torch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from ipdb import set_trace

IMG_ROOT = './data/VOCdevkit2007/VOC2007/JPEGImages/{}.jpg'

def vis_model_output(im_path, im_info, rois, bbox_pred, cfg):
    """
    for training phase:
        bbox_pred: (N*nr_rois, 4)
    """

    N = im_info.size(0)
    for i in range(N):
        path = im_path[i]
        info = im_info[i].unsqueeze(0)
        boxes = rois.data[i, :, 1:5].unsqueeze(0) # original box
        box_deltas = bbox_pred[i].data

        bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        bbox_normalize_stds = bbox_normalize_stds.cuda()
        bbox_normalize_means = bbox_normalize_means.cuda()

        box_deltas = box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means # (nr_rois, 4)
        box_deltas = box_deltas.view(1, -1, 4) # is that right?

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, info.data, 1)

        pred_boxes /= info.data[0][2] # data[1][0][2] # scale back
        pred_boxes = pred_boxes.squeeze()

        im2show = np.copy(cv2.imread(IMG_ROOT.format(path)))
        im2show = vis_detections(im2show, pred_boxes)
        cv2.imshow('test', im2show)
        if cv2.waitKey(0) == 27: # 'ESC'
            cv2.destroyAllWindows() 
            exit(0)

def vis_model_multioutput(im_path, im_info, rois, bbox_pred, b_rois, b_bbox_pred, cfg):
    """
    for training phase:
        bbox_pred: (N*nr_rois, 4)
    """

    N = im_info.size(0)
    for i in range(N):
        path = im_path[i]
        info = im_info[i].unsqueeze(0)
        boxes = rois.data[i, :, 1:5].unsqueeze(0) # original box
        box_deltas = bbox_pred[i].data
        b_boxes = b_rois.data[i, :, 1:5].unsqueeze(0)
        b_box_deltas = b_bbox_pred[i].data

        bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        bbox_normalize_stds = bbox_normalize_stds.cuda()
        bbox_normalize_means = bbox_normalize_means.cuda()

        box_deltas = box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means # (nr_rois, 4)
        box_deltas = box_deltas.view(1, -1, 4) # is that right?

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, info.data, 1)

        pred_boxes /= info.data[0][2] # data[1][0][2] # scale back
        pred_boxes = pred_boxes.squeeze()

        b_box_deltas = b_box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means # (nr_rois, 4)
        b_box_deltas = b_box_deltas.view(1, -1, 4) # is that right?

        b_pred_boxes = bbox_transform_inv(b_boxes, b_box_deltas, 1)
        b_pred_boxes = clip_boxes(b_pred_boxes, info.data, 1)

        b_pred_boxes /= info.data[0][2] # data[1][0][2] # scale back
        b_pred_boxes = b_pred_boxes.squeeze()     

        im2show = np.copy(cv2.imread(IMG_ROOT.format(path)))
        im2show = vis_detections(im2show, pred_boxes, color='green')
        im2show = vis_detections(im2show, b_pred_boxes, color='red')
        cv2.imshow('test', im2show)
        if cv2.waitKey(0) == 27: # 'ESC'
            cv2.destroyAllWindows() 
            exit(0)


def vis_detections(im, dets, class_name='', thresh=0.8, color='green'):
    """Visual debugging of detections."""
    dets = dets.cpu().numpy()
    # for i in range(np.minimum(10, dets.shape[0])):
    for i in range(dets.shape[0]):
        # set_trace()
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        if color == 'green':
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s' % i, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 204, 0), thickness=2)
        elif color == 'red':
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 204), 2)
            cv2.putText(im, '%s' % i, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 204), thickness=2)
        # score = dets[i, -1]
        # if score > thresh:
        #     cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            # cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
            #             1.0, (0, 0, 255), thickness=1)
    return im