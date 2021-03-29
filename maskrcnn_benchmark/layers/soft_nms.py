import numpy as np

def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep

# import numpy as np
# import torch
# import time 

# def area_of(left_top, right_bottom):

#     """Compute the areas of rectangles given two corners.
#     Args:
#         left_top (N, 2): left top corner.
#         right_bottom (N, 2): right bottom corner.
#     Returns:
#         area (N): return the area.
#         return types: torch.Tensor
#     """
#     hw = torch.clamp(right_bottom - left_top, min=0.0)
#     return hw[..., 0] * hw[..., 1]


# def iou_of(boxes0, boxes1, eps=1e-5):
#     """Return intersection-over-union (Jaccard index) of boxes.
#     Args:
#         boxes0 (N, 4): ground truth boxes.
#         boxes1 (N or 1, 4): predicted boxes.
#         eps: a small number to avoid 0 as denominator.
#     Returns:
#         iou (N): IoU values.
#     """
#     overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
#     overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

#     overlap_area = area_of(overlap_left_top, overlap_right_bottom)
#     area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
#     area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
#     return overlap_area / (area0 + area1 - overlap_area + eps)

# def softnms_cpu_torch(box_scores, score_threshold=0.001, sigma=0.5, top_k=-1):
#     """Soft NMS implementation.
#     References:
#         https://arxiv.org/abs/1704.04503
#         https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
#     Args:
#         box_scores (N, 5): boxes in corner-form and probabilities.
#         score_threshold: boxes with scores less than value are not considered.
#         sigma: the parameter in score re-computation.
#             scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
#         top_k: keep top_k results. If k <= 0, keep all the results.
#     Returns:
#          picked_box_scores (K, 5): results of NMS.
#     """
#     original_box = box_scores.clone()
#     picked_box_scores = []
#     picked_box_index = []
#     while box_scores.size(0) > 0:
#         max_score_index = torch.argmax(box_scores[:, 4])
#         picked_box_index.append(max_score_index)
#         cur_box_prob = box_scores[max_score_index, :].clone()
#         picked_box_scores.append(cur_box_prob)
#         if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
#             break
#         cur_box = cur_box_prob[:-1]
#         box_scores[max_score_index, :] = box_scores[-1, :]
#         box_scores = box_scores[:-1, :]
#         ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
#         box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
#         box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
#         # from ipdb import set_trace; set_trace()
#     if len(picked_box_scores) > 0:
#         return torch.stack(picked_box_scores), torch.tensor(picked_box_index)
#     else:
#         return torch.tensor([]), torch.tensor([], dtype=torch.long)



# from ipdb import set_trace; set_trace()
# from maskrcnn_benchmark.layers.nms_wrapper.cpu_nms import cpu_soft_nms
# import numpy as np

# def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

#     keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
#                         np.float32(sigma), np.float32(Nt),
#                         np.float32(threshold),
#                         np.uint8(method))
#     return keep
