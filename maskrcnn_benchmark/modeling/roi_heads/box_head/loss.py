# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss_weight

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False,
        has_weight=True,
        cfg=None,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.has_weight = has_weight
        self.cfg = cfg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal) # (nr_gt, nr_proposals)
        matched_idxs = self.proposal_matcher(match_quality_matrix) # (nr_proposals,)
        iou, _ = match_quality_matrix.max(dim=0) # (nr_proposals,)
        # from ipdb import set_trace; set_trace()
        # target.add_field("iou", iou)
        # Fast RCNN only need "labels" field for selecting the targets
        # target = target.copy_with_fields("labels")
        # @zk copy more items
        if target.has_field("scores"):
            if target.has_field("softlabels"):
                target = target.copy_with_fields(["labels", "scores", "is_gt", "softlabels"])
            else:
                target = target.copy_with_fields(["labels", "scores", "is_gt"])
        else:
            target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        matched_targets.add_field("iou", iou)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        weight_result = []
        ious = []
        softlabels = []
        is_gts = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            ) # 每个proposal match到的target
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            ious.append(matched_targets.get_field("iou"))
            if matched_targets.has_field('scores'):
                score_weight = matched_targets.get_field('scores')
                pred_mask = matched_targets.get_field('is_gt') != 1
                if self.cfg.MODEL.PSEUDO_CLASS_AGNOSTIC:
                    # set pred box‘s weight to 0.
                    base_weight = pred_mask.float() * 0. + \
                        (~pred_mask).float()
                else:
                    base_weight = pred_mask.float() * self.cfg.MODEL.PSEUDO_WEIGHT + \
                        (~pred_mask).float()
                weight_result.append(score_weight * base_weight)
                if matched_targets.has_field("softlabels"):
                    softlabels_per_image = matched_targets.get_field('softlabels')
                    softlabels.append(softlabels_per_image)
                    is_gts.append(matched_targets.get_field('is_gt'))
                # from ipdb import set_trace; set_trace()
        if targets[0].has_field('scores'):
            if targets[0].has_field('softlabels'):
                return labels, regression_targets, weight_result, ious, softlabels, is_gts
            else:
                return labels, regression_targets, weight_result, ious
        return labels, regression_targets, ious

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        if targets[0].has_field("scores"):
            if targets[0].has_field("softlabels"):
                labels, regression_targets, weight_result, ious, softlabels, is_gts = self.prepare_targets(proposals, targets)
            else:
                labels, regression_targets, weight_result, ious = self.prepare_targets(proposals, targets)
        else:
            labels, regression_targets, ious = self.prepare_targets(proposals, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        if targets[0].has_field("scores"):
            if targets[0].has_field("softlabels"):
                for labels_per_image, regression_targets_per_image, proposals_per_image, weight_per_image, iou_per_image, \
                    softlabel_per_image, is_gt_per_image \
                    in zip(labels, regression_targets, proposals, weight_result, ious, softlabels, is_gts
                ):
                    proposals_per_image.add_field("labels", labels_per_image)
                    proposals_per_image.add_field(
                        "regression_targets", regression_targets_per_image
                    )
                    proposals_per_image.add_field("weight", weight_per_image)
                    # from ipdb import set_trace; set_trace()
                    proposals_per_image.add_field("iou", iou_per_image)
                    proposals_per_image.add_field("softlabels", softlabel_per_image)
                    proposals_per_image.add_field("is_gts", is_gt_per_image)
            else:
                for labels_per_image, regression_targets_per_image, proposals_per_image, weight_per_image, iou_per_image in zip(
                    labels, regression_targets, proposals, weight_result, ious
                ):
                    proposals_per_image.add_field("labels", labels_per_image)
                    proposals_per_image.add_field(
                        "regression_targets", regression_targets_per_image
                    )
                    proposals_per_image.add_field("weight", weight_per_image)
                    # from ipdb import set_trace; set_trace()
                    proposals_per_image.add_field("iou", iou_per_image)
        else:
            for labels_per_image, regression_targets_per_image, proposals_per_image, iou_per_image in zip(
                labels, regression_targets, proposals, ious
            ):
                proposals_per_image.add_field("labels", labels_per_image)
                proposals_per_image.add_field(
                    "regression_targets", regression_targets_per_image
                )
                proposals_per_image.add_field("iou", iou_per_image)
        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # print('box_head | loss.py | class_logits size {0}'.format(class_logits[0].size()))
        # print('box_head | loss.py | box_regression size {0}'.format(box_regression[0].size()))
        class_logits = cat(class_logits, dim=0) # efficient version
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals
        # from ipdb import set_trace; set_trace()
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        if proposals[0].has_field("softlabels"):
            softlabels = cat([proposal.get_field("softlabels") for proposal in proposals], dim=0)
            is_gts = cat([proposal.get_field("is_gts") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        if proposals[0].has_field("weight") and self.has_weight:
            weight_result = cat([proposal.get_field("weight") for proposal in proposals], dim=0)
            # new_weight = (1-low_bound) / (1-confidence_treshold)*(weight-1) + 1
            weight_result = (weight_result - 1.) * \
                (1. - self.cfg.MODEL.LOW_BOUND) / (1. - self.cfg.MODEL.PSEUDO_CONF_THRESH) + 1.

        if proposals[0].has_field("softlabels"):
            # for gt cls, normal cross entropy loss
            # for pseudo cls, kl loss
            gt_mask = is_gts == 1.
            classification_loss = F.cross_entropy(class_logits[gt_mask], labels[gt_mask], reduction='none')
            pseudo_softlabels = softlabels[~gt_mask]
            num_softlabels = pseudo_softlabels.shape[1]
            pseudo_classlabels = F.softmax(class_logits[~gt_mask][:, :num_softlabels])
            if pseudo_softlabels.shape[0] == 0:
                pseudo_classification_loss = 0.
            else:
                pseudo_classification_loss = F.kl_div(torch.log(pseudo_classlabels), pseudo_softlabels)
                if self.cfg.MODEL.SOFT_AUTO_WEIGHT:
                    gt_num = gt_mask.sum()
                    pseudo_num = gt_mask.size(0) - gt_num
                    auto_weight = torch.clamp(gt_num.float() / pseudo_num.float(), 0., 20.)
                    pseudo_classification_loss *= auto_weight
                    print(gt_num.item(), pseudo_num.item(), auto_weight.item())
                    # pseudo_classification_loss *= gt_mask.sum
        else:
            # classification_loss = F.cross_entropy(class_logits, labels)
            classification_loss = F.cross_entropy(class_logits, labels, reduction='none')
            if proposals[0].has_field("weight") and self.has_weight:
                classification_loss = classification_loss * weight_result

        classification_loss = torch.mean(classification_loss)
        # get indices that correspond to the regression targets for the corresponding ground truth labels, to be used
        # with advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg: # F 只分辨含目标与不含目标两类
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else: # 获得含有目标的边框在对应box_regression中的索引
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)
        # from ipdb import set_trace; set_trace()

        if proposals[0].has_field("weight") and self.has_weight:
            box_loss = smooth_l1_loss_weight(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,  # sum
                beta=1,
                weight=weight_result[sampled_pos_inds_subset],
            )
        else:
            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,  # sum
                beta=1,
            )
        box_loss = box_loss / labels.numel()

        if proposals[0].has_field("softlabels"):
            return classification_loss, box_loss, pseudo_classification_loss
        else:
            return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    ) # 用于匹配proposal和gt box

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights) # 用于对box编码成regressable的格式

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    ) # 用于确保pos/neg的比例

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    has_weight = cfg.MODEL.ROI_HEADS.HAS_WEIGHT

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg,
        has_weight,
        cfg,
    )

    return loss_evaluator
