# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import smooth_l1_loss_weight
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func, has_weight=True, cfg=None):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler # BalancedPositiveNegativeSampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func # 标签生成函数，用以生成锚点对应的基准边框的索引: generate_rpn_labels
        self.discard_cases = ['not_visibility', 'between_thresholds'] # 指定需要放弃的锚点类型
        self.has_weight = has_weight
        self.cfg = cfg

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        # print('rpn | loss.py | match_targets_to_anchors | anchor : {0}'.format(anchor))
        # print('rpn | loss.py | match_targets_to_anchors | target : {0}'.format(target))
        # 计算所有anchor与所有gt之间的IoU
        match_quality_matrix = boxlist_iou(target, anchor)  # [num of target box, num of bounding box]
        # print('rpn | loss.py | match_targets_to_anchors | match_quality_matrix size : {0}'.format(match_quality_matrix.size()))

        # value = 0 ~ (M-1) means which gt to match to
        # value = -1 or -2 means no gt to match to, -1 = below_low_threshold, -2 = between_thresholds
        matched_idxs = self.proposal_matcher(match_quality_matrix)  # [num of bounding box]
        # print('rpn | loss.py | match_targets_to_anchors | matched_idxs size : {0}'.format(matched_idxs.size()))

        # RPN doesn't need any fields from target for creating the labels, so clear them all by createing a BoxList without empty fields
        ## comment below line to keep all the fields
        # target = target.copy_with_fields(copied_fields)

        # get the targets corresponding GT for each anchor
        # need to clamp the indices because we can have a single GT in the image, and matched_idxs can be -2, which goes out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        # print('rpn | loss.py | match_targets_to_anchors | matched_targets : {0}'.format(matched_targets))

        matched_targets.add_field("matched_idxs", matched_idxs)
        # from ipdb import set_trace; set_trace()
        return matched_targets, match_quality_matrix
    """
    - 获得锚点(anchor)的标签：-1为要舍弃的，０为背景，其余的为对应的gt。
    - 获得所有锚点与和其对应的gt的偏差，即边框回归
    """
    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        overlap_result = []
        matched_result = []
        weight_result = []
        for anchors_per_image, targets_per_image in zip(anchors, targets): # 循环从每一张图片中读取anchor和gt,然后进行处理
            # 得到与各个锚点对应的gt, 所有anchor与所有gt之间的IoU matrix
            matched_targets, matched_quality_matrix = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields)
            # 得到与各个锚点对应的gt的索引
            matched_idxs = matched_targets.get_field("matched_idxs")

            # 得到与各个锚点对应的gt的标签列表，其中０为舍弃，１为有用边框
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples) 得到与各个锚点内容为背景的索引，并将其标签设为０
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD  # match_idxs = -1 means below_low_threshold
            labels_per_image[bg_indices] = 0  # make these anchors' label to be 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases: # 处理超出图片的锚点
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1  # make these anchors' label to be -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases: # 丢掉IoU介于背景和目标之间的锚点
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1  # make these anchors' label to be -1

            # print('rpn | loss.py | prepare_targets | labels_per_image size : {0}'.format(labels_per_image.size()))

            # compute regression targets 计算每张图片中，所有锚点与其对应基准边框之间的偏差
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)
            # 将标签信息和边框回归信息保存到最开始初始化的列表里
            labels.append(labels_per_image) # -1,0,1 labels
            regression_targets.append(regression_targets_per_image)
            overlap_result.append(matched_quality_matrix)
            matched_result.append(matched_idxs)
            # 如果在入口文件中 给BoxList添加了scores
            if matched_targets.has_field('scores'):
                weight_result.append(matched_targets.get_field('scores'))
            # set_trace()
        if matched_targets.has_field('scores'):
            return labels, regression_targets, overlap_result, matched_result, weight_result
        return labels, regression_targets, overlap_result, matched_result

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor]) 由FPN得到的计算目标得分的特征图
            box_regression (list[Tensor]) 由FPN得到的计算边框回归的特征图
            targets (list[BoxList]) 每个图片上的边框(gt)

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors] # 分别将每一个图片的不同FPN层中生成的锚点合并起来
        #labels, regression_targets, overlap_result, matched_result = self.prepare_targets(anchors, targets)
        targets_result = self.prepare_targets(anchors, targets)
        if len(targets_result) == 4:
            labels, regression_targets, overlap_result, matched_result = targets_result
        else:
            labels, regression_targets, overlap_result, matched_result, weight_result = targets_result

        # print('rpn | loss.py | call | labels size : {0}'.format(labels[0].size()))
        # print('rpn | loss.py | call | regression_targets size : {0}'.format(regression_targets[0].size()))
        # print('rpn | loss.py | call | overlap_result size : {0}'.format(overlap_result[0].size()))
        # print('rpn | loss.py | call | matched_result size : {0}'.format(matched_result[0].size()))
        # 根据所有锚点的标签选取作为背景的锚点和作为目标的锚点的标签，该标签中0为未选择，１为选择
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels) # list[tensor], list[tensor] list的长度代表图片的个数
        # from ipdb import set_trace; set_trace()
        # print('rpn | loss.py | call | sampled_pos_inds size : {0}'.format(sampled_pos_inds[0].size()))
        # print('rpn | loss.py | call | sampled_neg_inds size : {0}'.format(sampled_neg_inds[0].size()))
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1) # 将selected的anchor的index返回
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        # print('rpn | loss.py | call | sampled_pos_inds size : {0}'.format(sampled_pos_inds.size()))
        # print('rpn | loss.py | call | sampled_pos_inds : {0}'.format(sampled_pos_inds))
        # print('rpn | loss.py | call | sampled_neg_inds size : {0}'.format(sampled_neg_inds.size()))
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0) # 将选中的正负锚点索引值合并到一起
        # print('rpn | loss.py | call | sampled_inds size : {0}'.format(sampled_inds.size()))

        sampled_inds_overlap = torch.zeros(sampled_inds.size())
        # print('rpn | loss.py | call | sampled_inds_overlap size : {0}'.format(sampled_inds_overlap.size()))
        count = 0
        """
        ### add by Faster-ILOD ####
        # !there mines some bug when ims_per_gpu > 1
        # from ipdb import set_trace; set_trace()
        for index in sampled_inds:
            try:
                _ = labels[0][index] # len(labels) == ims_per_gpu, 所以不应该直接用0！！！，这是有错误的
            except:
                from ipdb import set_trace; set_trace()
                print("Somthing wrong with sampled_inds and labels")
            label = labels[0][index]  # 1 = positive proposal, 0 = negative proposal, -1 = useless proposal
            if label > 0:
                matched_gt = matched_result[0][index] # matched_result：各个anchor对应的gt的索引(-1表示没有匹配到gt)
                overlap = overlap_result[0][matched_gt][index] # 得到该anchor与对应的gt在anchor-gt的IOU matrix中的IOU值
                # overlap = overlap.to('cpu')
                # overlap = float(overlap.data.numpy())
                # print('index = {0}, matched_gt = {1}, overlap = {2}'.format(index, matched_gt, overlap))
            elif label == 0:
                overlap = 0
            else:
                raise ValueError('Something goes wrong at proposal choosing procedure.')
            sampled_inds_overlap[count] = overlap
            count = count + 1
        sampled_inds_overlap = sampled_inds_overlap.to('cuda')
        # print('rpn | loss.py | call | sampled_inds_overlap : {0}'.format(sampled_inds_overlap))

        scaled_weight = -50 * torch.exp(-20 * sampled_inds_overlap) # 用IOU来作为weight加权，但是最后没有用到loss上。。。
        scaled_weight = 0.25 + 0.75 * torch.exp(scaled_weight)  # default a = 0.25
        # print('rpn | loss.py | call | scaled_weight : {0}'.format(scaled_weight))
        # final_scaled_weight = scaled_weight.clone()
        # final_scaled_weight[sampled_inds_overlap >= 0.5] = 1
        # print('rpn | loss.py | call | final_scaled_weight : {0}'.format(final_scaled_weight))
        """
        # 将所有图片中的RPN　Head中的边框目标得分层和边框回归层分别合并成统一张量N*ratio（或N*４ratio）的边框分类信息和边框回归信息
        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        if len(targets_result) == 4 or (not self.has_weight):
            box_loss = smooth_l1_loss(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1.0/9, size_average=False) / (sampled_inds.numel())
        else: # 返回了weight，需要计算weight loss
            weight_result = torch.cat(weight_result, dim=0)
            # new_weight = (1-low_bound) / (1-confidence_treshold)*(weight-1) + 1
            weight_result = (weight_result - 1.) * \
                (1. - self.cfg.MODEL.LOW_BOUND) / (1. - self.cfg.MODEL.PSEUDO_CONF_THRESH) + 1.
            box_loss = smooth_l1_loss_weight(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1.0/9, size_average=False, weight=weight_result[sampled_pos_inds]) / (sampled_inds.numel())
            
        # print('rpn | loss.py | call | box_loss : {0}'.format(box_loss))
        # from ipdb import set_trace; set_trace()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], weight=None, size_average=None, reduce=None, reduction='none')
        if len(targets_result) == 5 and self.has_weight:
            objectness_loss = objectness_loss * weight_result[sampled_inds]
        original_objectness_loss = torch.mean(objectness_loss)

        # print('rpn | loss.py | call | original_objectness_loss : {0}'.format(original_objectness_loss))
        
        # scaled_objectness_loss = objectness_loss * scaled_weight
        # scaled_objectness_loss = torch.mean(scaled_objectness_loss)
        # print('rpn | loss.py | call | scaled_objectness_loss : {0}'.format(scaled_objectness_loss))
        return original_objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0  # if >=0 means having gt, label = 1; else label = 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    has_weight = cfg.MODEL.RPN.HAS_WEIGHT
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder, generate_rpn_labels, has_weight, cfg)
    return loss_evaluator
