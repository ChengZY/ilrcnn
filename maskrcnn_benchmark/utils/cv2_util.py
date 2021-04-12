"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2
from torchvision.transforms import functional as F
import numpy as np
import torch

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

CATEGORIES = ["__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def vis_pred(images, targets, src_pred):
    # images: ImageList images.tensors.shape, images.image_sizes
    for k in range(images.tensors.shape[0]):
        # transform from tensor-> array
        cv2_img = F.normalize(images.tensors[k], mean=[-102.9801/1., -115.9465/1., -122.7717/1.], std=[1., 1., 1.])
        cv2_img = np.array(cv2_img.permute(1,2,0).cpu()).astype(np.uint8)
        h, w = cv2_img.shape[:-1]
        print(cv2_img.shape)
        # draw gt box on array
        for j in range(targets[k].bbox.shape[0]): # travel each box in one img
            bbox = targets[k].bbox[j].cpu()
            cv2.rectangle(cv2_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

        ss = src_pred[k].get_field("scores")
        keep = torch.nonzero(ss > 0.7).squeeze(1)
        src_pred[k] = src_pred[k][keep]

        src_score = src_pred[k].get_field("scores").tolist()
        src_label = src_pred[k].get_field("labels").tolist()
        src_bbox = src_pred[k].resize((w, h)).bbox.data.cpu()
        src_catname = [CATEGORIES[i] for i in src_label]

        s = "{}: {:.2f}"
        # overlap src prediciton box
        for m in range(src_bbox.shape[0]):
            cv2.rectangle(cv2_img, (src_bbox[m][0], src_bbox[m][1]), (src_bbox[m][2], src_bbox[m][3]), (0,0,255), 2)
            cv2.putText(cv2_img, s.format(src_catname[m], src_score[m]), (src_bbox[m][0], src_bbox[m][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)
        cv2.imshow("VOC", cv2_img)
        if cv2.waitKey(0) == 27: # 'ESC'
            cv2.destroyAllWindows() 
            exit(0)