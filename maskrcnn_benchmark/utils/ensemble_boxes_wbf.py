# coding: utf-8

import warnings
import numpy as np
from torchvision.transforms import functional as F
import cv2

# print("Import numba...")
# from numba import jit
# print("Import numba successfully")
# @jit(nopython=True)
def bb_intersection_over_union(A, B): #-> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)): # 对每个model
        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])): # 每个model里的每个box
            # from ipdb import set_trace; set_trace()
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            # if x2 < x1:
            #     warnings.warn('X2 < X1 value in box. Swap them.')
            #     x1, x2 = x2, x1
            # if y2 < y1:
            #     warnings.warn('Y2 < Y1 value in box. Swap them.')
            #     y1, y2 = y2, y1
            # if x1 < 0:
            #     warnings.warn('X1 < 0 in box. Set it to 0.')
            #     x1 = 0
            # if x1 > 1:
            #     warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            #     x1 = 1
            # if x2 < 0:
            #     warnings.warn('X2 < 0 in box. Set it to 0.')
            #     x2 = 0
            # if x2 > 1:
            #     warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            #     x2 = 1
            # if y1 < 0:
            #     warnings.warn('Y1 < 0 in box. Set it to 0.')
            #     y1 = 0
            # if y1 > 1:
            #     warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            #     y1 = 1
            # if y2 < 0:
            #     warnings.warn('Y2 < 0 in box. Set it to 0.')
            #     y2 = 0
            # if y2 > 1:
            #     warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            #     y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue
            # from ipdb import set_trace; set_trace()
            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)
    # from ipdb import set_trace; set_trace()
    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes: # k \in label
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]] # 根据score降序排序

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1 # model index field is retained for consistensy but is not used.
    box[4:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)): # 对weighted box里面的box遍历
        box = boxes_list[i]
        if box[0] != new_box[0]: # label不一样。。
            continue
        iou = bb_intersection_over_union(box[4:], new_box[4:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()
    # filtered_boxes: dict(class_label: np.array(x)) x: np.array([[label, score * weights, weights, model_idx, x1, y1, x2, y2],...])
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    # from ipdb import set_trace; set_trace()
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j]) # 加入到new_boxes簇中
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type) # 根据簇计算weighted_box并更新weighted_box list
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())
        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)): # 每个簇
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(clustered_boxes) / weighted_boxes[i][2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i][1] = weighted_boxes[i][1] *  clustered_boxes[idx, 2].sum() / weights.sum()

            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(clustered_boxes) / (weighted_boxes[i][2] + weights[mask].sum())
            elif not allows_overflow: # T
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(clustered_boxes)) / weights.sum() # 调整该簇的weighted box的score
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes)) # 把融合后的box加入到最终的list中
    overall_boxes = np.concatenate(overall_boxes, axis=0) # concat所有label的weighted box
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]] # 降序排序
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

MEAN=[-102.9801/1., -115.9465/1., -122.7717/1.]
STD=[1., 1., 1.]
CATEGORIES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "person"]

# add by zhengkai
def visual_ens_box(image, target):
    cv2_img = F.normalize(image, mean=MEAN, std=STD)
    cv2_img = np.array(cv2_img.permute(1,2,0).cpu()).astype(np.uint8)
    h, w = cv2_img.shape[:-1]
    for j in range(target.bbox.shape[0]):
        bbox = target.bbox[j].cpu()
        score = target.get_field("scores")[j]
        label = target.get_field("labels")[j]
        catname = CATEGORIES[label]
        s = "{}: {:.2f}"
        if score == 1.: # is gt, then GREEN
            cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
        else: # is pseudo box
            cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
            cv2.putText(cv2_img, s.format(catname, score), (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)
        # from ipdb import set_trace; set_trace()
        # cv2.imshow("After Ensemble", cv2_img)
        # if cv2.waitKey(0) == 27: # 'ESC'
        #     cv2.destroyAllWindows() 
        #     exit(0)
    return cv2_img

ITER_RGB = [(255,255,255), (184,244,206), (95,226,142), (2,147,54), (0,0,0)]

def visual_multi_iter_box(image, boxlist, scorelist, labellist, iterlist):
    cv2_img = F.normalize(image, mean=MEAN, std=STD)
    cv2_img = np.array(cv2_img.permute(1,2,0).cpu()).astype(np.uint8)
    h, w = cv2_img.shape[:-1]
    for i, iter_idx in enumerate(iterlist):
        boxes = boxlist[i]
        scores = scorelist[i]
        labels = labellist[i]
        print("Iter: ", iter_idx, " Boxes num: ", len(boxes))
        for j in range(len(boxes)):
            bbox = boxes[j]
            score = scores[j]
            label = labels[j]
            catname = CATEGORIES[label]
            s = "{}: {:.2f}"
            cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), ITER_RGB[i], 2)
            cv2.putText(cv2_img, s.format(catname, score), (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)

    # cv2.imshow("Before Ensemble", cv2_img)

    # if cv2.waitKey(0) == 27: # 'ESC'
    #     cv2.destroyAllWindows() 
    #     exit(0)
    return cv2_img