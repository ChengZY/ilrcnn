# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_predictors import make_roi_box_predictor
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import make_roi_box_post_processor
from maskrcnn_benchmark.modeling.roi_heads.box_head.loss import make_roi_box_loss_evaluator

class PrototypePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypePredictor, self).__init__()
        hidden_dim = config.MODEL.HIDDEN_DIM
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        K = config.MODEL.QLEN

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, hidden_dim),
        )         
        # self.fc = nn.Linear(in_channels, hidden_dim)

        # self.projection = nn.Sequential(self.fc, nn.ReLU())

        # create the queue
        # self.register_buffer("queue", torch.randn(num_classes, hidden_dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=1)
        # self.register("queue_ptr", torch.zeros((num_classes,), dtype=torch.long))
        """
        先设置一种简单的contrastive loss，只计算当前nr_proposal中的contrastive
        看unqiue label，如果只有一个unique label，则用背景类代替推远，否则用正常类推远
        """


    def forward(self, x):
        """
        - x: (nr_proposals, c, h, w)
        - proposals: list(BoxList)
        """
        x = self.avgpool(x) # squeeze h,w to 1
        x = x.view(x.size(0), -1) # (nr_proposals, c)
        emb = self.projection(x) # (nr_porposals, dim)
        norm_emb = nn.functional.normalize(emb, dim=1)

        # concat all labels in list(BoxList) for multiple image case
        # labels = torch.cat([x.get_field('labels') for x in proposals], dim=0)
        # assert emb.shape[0] == labels.shape[0]

        return norm_emb
        # enqueue
        # with torch.no_grad():
        #     ulabels = torch.unique(labels)
        #     for label in ulabels:
        #         ptr = int(self.queue_ptr[label])
        #         mask = labels == label
        #         self.queue[label, :, ]emb[mask]

        # """
        # 在当前batch size内contrastive
        # """
        # ulabels = torch.unique(labels)
        # for label in ulabels:
        #     if label == 0:
        #         continue
            
        #     is_mask = ulabels == label
        #     isnot_mask = ulabels != label
        # from ipdb import set_trace; set_trace()
        # return labels


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_type='none', lamb=0.5):
        """
        Args;
            temperature: to enlarge the similarity magnitude
            iou_threshold: only compute contrastive loss for the prediction has high iou with gt box
        """
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_type = reweight_type
        self.lamb = lamb
    
    def forward(self, features, labels, ious):
        """
        Args;
            features [tensor]; (nr_rois, dim)
            labels [tennsor]; (nr_rois)
            ious [tensor]; (nr_rois)
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        # mask_{i,j} = 1 means R_i and R_j have the same label
        label_mask = torch.eq(labels[:,None], labels[:,None].t()).float()
        similarity = torch.matmul(features, features.t()) / self.temperature
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()
        # mask out self
        # self_mask = torch.ones_like(similarity)
        # self_mask.fill_diagonal_(0)
        self_mask = (~torch.eye(similarity.shape[0], dtype=torch.uint8)).float().cuda()

        exp_sim = torch.exp(similarity) * self_mask
        # log(e^x/\sum{e^y}) = x - log(\sum{e^y})
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * self_mask * label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
        # compute non-background mask
        # 对background label的不需要计算contrastive loss
        pos_keep = labels != 0
        keep = keep & pos_keep
        per_label_log_prob = per_label_log_prob[keep]

        loss = -per_label_log_prob

        coef = self._get_weight_func(self.reweight_type)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean() * self.lamb
    
    def _get_weight_func(self, wtype):
        def non(iou):
            return torch.ones_like(iou)
        def iden(iou):
            return iou

        if wtype == 'none':
            return non
        elif wtype == 'identity':
            return iden

class ROIProtoBoxHead(torch.nn.Module):
    """
    Generic Box Head + Prototype Head class
    """

    def __init__(self, cfg, in_channels):
        super(ROIProtoBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.proto_predictor = PrototypePredictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.conloss_evaluator = SupConLoss(
            cfg.MODEL.TEMPERATURE, cfg.MODEL.CONTRAST_IOU_THRES, 
            cfg.MODEL.REWEIGHT_TYPE, cfg.MODEL.LAMB_CONTRAST)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets) # @zk add "iou" item
        # from ipdb import set_trace; set_trace()
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        # features: list([N,C,H,W]), proposals: list(BoxList) x: (nr_proposals,c,h,W)
        x = self.feature_extractor(features, proposals)
        # from ipdb import set_trace; set_trace()
        # final classifier that converts the features into predictions
        ##### Original Branch #####
        class_logits, box_regression = self.predictor(x) # two images's concat result output
        ##### Contrastive Branch #####
        box_cls_norm = self.proto_predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])
        # concat all labels in list(BoxList) for multiple image case
        labels = torch.cat([x.get_field('labels') for x in proposals], dim=0)
        ious = torch.cat([x.get_field('iou') for x in proposals], dim=0)
        # from ipdb import set_trace; set_trace()
        loss_contrast = self.conloss_evaluator(box_cls_norm, labels, ious)
        return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_contrast=loss_contrast)


def build_roi_proto_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIProtoBoxHead(cfg, in_channels)
