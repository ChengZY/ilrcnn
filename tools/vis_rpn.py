# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import cv2
from PIL import Image

from maskrcnn_benchmark.config import cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from maskrcnn_benchmark.data import make_data_loader  # import data set
from maskrcnn_benchmark.engine.inference import inference  # inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict  # when multiple gpus are used, reduce the loss
from maskrcnn_benchmark.modeling.detector import build_detection_model  # used to create model
from maskrcnn_benchmark.solver import make_lr_scheduler  # learning rate updating strategy
from maskrcnn_benchmark.solver import make_optimizer  # setting the optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank  # related to multi-gpu training; when usong 1 gpu, get_rank() will return 0
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger  # related to logging model(output training status)
from maskrcnn_benchmark.utils.miscellaneous import mkdir  # related to folder creation
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from tensorboardX import SummaryWriter
from maskrcnn_benchmark.distillation.distillation import calculate_rpn_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_feature_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_roi_distillation_losses
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from maskrcnn_benchmark.structures.bounding_box import BoxList
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from collections import Counter
import matplotlib.pyplot as plt
import PIL
import io
CATEGORIES = ["__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

CLS_SETS = {
    'v1010': (("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow"),
            ("diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")),
    'v155': (("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
            "diningtable", "dog", "horse", "motorbike", "person"),
            ("pottedplant", "sheep", "sofa", "train", "tvmonitor")),
    'v191': (("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train"),
            ("tvmonitor",))
}

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

def stat_pred(images, src_pred):
    """
    统计pred的instance的分布
    """
    pred_cat = []
    # images: ImageList images.tensors.shape, images.image_sizes
    for k in range(images.tensors.shape[0]):
        # src_score = src_pred[k].get_field("scores").tolist()
        src_label = src_pred[k].get_field("labels").tolist()
        # src_bbox = src_pred[k].resize((w, h)).bbox.data.cpu()
        src_catname = [CATEGORIES[i] for i in src_label]

        pred_cat.extend(src_catname)

    return pred_cat


def do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_source, checkpointer_target,
             device, checkpoint_period, arguments_source, arguments_target, summary_writer, cfg_target):

    # record log information
    logger = logging.getLogger("maskrcnn_benchmark_target_model.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")  # used to record
    max_iter = len(data_loader)  # data loader rewrites the len() function and allows it to return the number of batches (cfg.SOLVER.MAX_ITER)
    print("++++++" * 5)
    print(max_iter)
    start_iter = arguments_target["iteration"]  # 0
    model_target.train()  # set the target model in training mode
    model_source.eval()  # set the source model in inference mode
    start_training_time = time.time()
    end = time.time()
    average_distillation_loss = 0
    average_faster_rcnn_loss = 0

    if args.stat:
        pred_catlist = []
    # from ipdb import set_trace; set_trace()
    # for iteration, (images, targets, x, idx, im_id) in enumerate(data_loader, start_iter):
    for iteration, (images, targets, _, idx) in enumerate(data_loader, start_iter):
        # from ipdb import set_trace; set_trace()
        data_time = time.time() - end
        iteration = iteration + 1
        arguments_target["iteration"] = iteration
        scheduler.step()  # update the learning rate

        images = images.to(device)   # move images to the device
        targets = [target.to(device) for target in targets]  # move targets (labels) to the device
        
        # source model prediction
        src_pred = model_source(images)[0]

        # 可视化过滤pred box之前的所有pred box
        # if args.vis:
        #     vis_pred(images, targets, src_pred)
        #     continue

        #### 添加pred到gt中 ####
        # per image
        # for gt, pred in zip(targets, src_pred):
        #     before_num = gt.bbox.shape[0]
        #     """
        #     targets: BoxList -> bbox, size, mode, extra_fields, difficult
        #     """
        #     pred_scores = pred.get_field("scores")
        #     keep = torch.nonzero(pred_scores > cfg_target.MODEL.PSEUDO_CONF_THRESH).squeeze(1)
        #     pred = pred[keep]
            
        #     with torch.no_grad():
        #         if len(pred) > 0:
        #             iou_matrix = boxlist_iou(pred, gt)
        #             max_iou = iou_matrix.max(dim=1)[0]
        #             # set_trace()
        #             keep2 = torch.nonzero(max_iou < cfg_target.MODEL.PSEUDO_IOU_THRESH).squeeze(1)
        #             pred = pred[keep2]

        #     pred_scores = pred.get_field("scores")# .tolist() # update pred scores
        #     pred_labels = pred.get_field("labels") # .tolist() # update pred labels
        #     pred_boxes = pred.bbox.data # update pred bbox

        #     merge_boxes = torch.cat([gt.bbox, pred_boxes], dim=0)
        #     gt.bbox = merge_boxes
        #     gt.add_field('scores', torch.cat([torch.ones_like(gt.extra_fields['labels'], dtype=torch.float32), pred_scores], dim=0))
        #     gt.add_field('is_gt', torch.cat([torch.ones_like(gt.extra_fields['labels']), torch.zeros_like(pred_labels)], dim=0))
        #     gt.extra_fields['labels'] = torch.cat([gt.extra_fields['labels'], pred_labels], dim=0)
        #     gt.extra_fields['difficult'] = torch.cat([gt.extra_fields['difficult'], torch.zeros_like(pred_labels).to(torch.uint8)], dim=0)
        #     after_num = gt.bbox.shape[0]
        #     print("Box Number transformation: {} -> {}".format(before_num, after_num))

        # 可视化过滤pred box之后的pred box
        # if args.vis:
        #     # vis_pred(images, targets, src_pred)
        #     for k in range(images.tensors.shape[0]):
        #         # transform from tensor-> array
        #         cv2_img = F.normalize(images.tensors[k], mean=[-102.9801/1., -115.9465/1., -122.7717/1.], std=[1., 1., 1.])
        #         cv2_img = np.array(cv2_img.permute(1,2,0).cpu()).astype(np.uint8)
        #         h, w = cv2_img.shape[:-1]
        #         # print(cv2_img.shape)
        #         # draw gt box on array
        #         for j in range(targets[k].bbox.shape[0]): # travel each box in one img
        #             bbox = targets[k].bbox[j].cpu()
        #             score = targets[k].get_field("scores")[j]
        #             label = targets[k].get_field("labels")[j]
        #             catname = CATEGORIES[label]
        #             s = "{}: {:.2f}"
        #             if score == 1.: # is gt, then GREEN
        #                 cv2.rectangle(cv2_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        #             else:
        #                 cv2.rectangle(cv2_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        #                 cv2.putText(cv2_img, s.format(catname, score), (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)
        #         # from ipdb import set_trace; set_trace()
        #         cv2.imshow("VOC", cv2_img)
        #         if cv2.waitKey(0) == 27: # 'ESC'
        #             cv2.destroyAllWindows() 
        #             exit(0)
        #     continue

        if args.stat:
            tmp = stat_pred(images, src_pred)
            pred_catlist.extend(tmp)
            continue

        loss_dict_target, feature_target, backbone_feature_target, anchor_target, rpn_output_target = model_target(images, targets)
        faster_rcnn_losses = sum(loss for loss in loss_dict_target.values())  # summarise the losses for faster rcnn

        roi_distillation_losses, rpn_output_source, feature_source, backbone_feature_source, soften_result, soften_proposal, feature_proposals \
            = calculate_roi_distillation_losses(model_source, model_target, images)
        ### vis top64 rpn proposals ####
        """
        soften_result: 
        soften_proposal: 
        """
        # from ipdb import set_trace; set_trace()
        if args.vis:
            ROOT = "/home/zhengkai/maskrcnn-benchmark/datasets/voc/VOC2007"
            annopath = os.path.join(ROOT, "Annotations", "%s.xml")
            imgpath = os.path.join(ROOT, "JPEGImages", "%s.jpg")
            (soften_scores, soften_bboxes) = soften_result
            # from ipdb import set_trace; set_trace()
            # using FastRCNNLossComputation to matched the GT
            idxs = list(idx)
            matched_labels, _ = model_target.roi_heads.box.loss_evaluator.prepare_targets(soften_proposal, targets)
            for k in range(images.tensors.shape[0]):
                im_id = data_loader.dataset.get_img_id(idxs[k])
                # transform from tensor-> array
                cv2_img = F.normalize(images.tensors[k], mean=[-102.9801/1., -115.9465/1., -122.7717/1.], std=[1., 1., 1.])
                cv2_img = np.array(cv2_img.permute(1,2,0).cpu()).astype(np.uint8)
                h, w = cv2_img.shape[:-1]

                # draw gt box on array
                for j in range(targets[k].bbox.shape[0]): # travel each box in one img
                    bbox = targets[k].bbox[j].cpu()
                    # score = targets[k].get_field("scores")[j]
                    label = targets[k].get_field("labels")[j]
                    catname = CATEGORIES[label]
                    if catname in data_loader.dataset.old_classes:
                        color = (255,0,0)
                    else:
                        color = (0,255,0)
                    cv2.rectangle(cv2_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(cv2_img, catname, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1., color, 1)

                # draw rpn proposal on array
                for j in range(soften_proposal[k].bbox.shape[0]): # travel each box in one img
                    # 注意这个box是regression之前的proposal，并不是最终输出，所以没完全拟合到gt box上
                    bbox = soften_proposal[k].bbox[j].cpu()
                    match_label = matched_labels[k][j]
                    # from ipdb import set_trace; set_trace()
                    # score = soften_proposal[k].get_field("objectness")[j]
                    soft_scores = soften_scores[j]
                    soft_boxes = soften_bboxes[j]
                    score = soft_scores.max().item()
                    pred_label = soft_scores.argmax().item()
                    col_idx = list(range(len(data_loader.dataset.old_classes)+1))
                    row_idx = soft_scores.data.cpu().numpy().tolist()
                    softmax_row_idx = torch.nn.functional.softmax(soft_scores).data.cpu().numpy().tolist()
                    fig = plt.figure()
                    ax = fig.add_axes([0.1,0.1,0.8,0.6])
                    ax.plot(col_idx, row_idx, 'r')
                    ax.plot(col_idx, softmax_row_idx, 'b')
                    plt.xticks(col_idx, data_loader.dataset.old_classes, rotation = 45)
                    ax.set_title('old class logits')
                    row_idx.sort(reverse=True)
                    print("logits: ", row_idx)
                    print("softmax: ", softmax_row_idx)

                    # buffer application
                    buffer_ = io.BytesIO()
                    fig.savefig(buffer_, format='png')
                    buffer_.seek(0)
                    pil_img = PIL.Image.open(buffer_)
                    ar = np.asarray(pil_img)

                    cv2.imshow("old class logits", ar)
                    distill_catname = CATEGORIES[pred_label]
                    rcnn_catname = CATEGORIES[match_label]
                    distill_s = "{}: {:.2f}"
                    rcnn_s = "{}"
                    cv2_img_copy = cv2_img.copy()
                    cv2.rectangle(cv2_img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
                    cv2.putText(cv2_img_copy, distill_s.format(distill_catname, score), (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)
                    cv2.putText(cv2_img_copy, rcnn_s.format(rcnn_catname), (bbox[0], bbox[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)
                    # count on proposals/total_proposals
                    cv2.putText(cv2_img_copy, "{}/{}".format(j, soften_proposal[k].bbox.shape[0]), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 1)
                    cv2.imshow("Image", cv2_img_copy)
                    if cv2.waitKey(0) == 27: # 'ESC'
                        buffer_.close()
                        cv2.destroyAllWindows() 
                        exit(0)
                    buffer_.close()

                # cv2.imshow("VOC", cv2_img)
                # if cv2.waitKey(0) == 27: # 'ESC'
                #     buffer_.close()
                #     cv2.destroyAllWindows() 
                #     exit(0)
            continue
        """
        rpn_distillation_losses = calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss='filtered_l2', bbox_loss='l2', bbox_threshold=0.1)

        feature_distillation_losses = calculate_feature_distillation_loss(feature_source, feature_target, loss='normalized_filtered_l1')

        distillation_losses = roi_distillation_losses + rpn_distillation_losses + feature_distillation_losses

        distillation_dict = {}
        distillation_dict['distillation_loss'] = distillation_losses.clone().detach()
        loss_dict_target.update(distillation_dict)
        """
        # print('loss_dict_target: {0}'.format(loss_dict_target))

        # losses = (faster_rcnn_losses * 1 + distillation_losses * 19)/20
        losses = faster_rcnn_losses#  + distillation_losses

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict_target)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if (iteration - 1) > 0:
            # average_distillation_loss = (average_distillation_loss * (iteration - 1) + distillation_losses) / iteration
            average_faster_rcnn_loss = (average_faster_rcnn_loss * (iteration - 1) + faster_rcnn_losses) /iteration
        else:
            # average_distillation_loss = distillation_losses
            average_faster_rcnn_loss = faster_rcnn_losses

        optimizer.zero_grad()  # clear the gradient cache
        # If mixed precision is not used, this ends up doing nothing, otherwise apply loss scaling for mixed-precision recipe.
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()  # use back-propagation to update the gradient
        optimizer.step()  # update learning rate

        # time used to do one batch processing
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # according to time'moving average to calculate how much time needed to finish the training
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        # for every 50 iterations, display the training status
        if iteration % 20 == 0 or iteration == max_iter:
            print('enter logger info')
            logger.info(
                meters.delimiter.join(["eta: {eta}", "iter: {iter}", "{meters}", "lr: {lr:.6f}", "max mem: {memory:.0f}"
                                       ]).format(eta=eta_string, iter=iteration, meters=str(meters),
                                                 lr=optimizer.param_groups[0]["lr"],
                                                 memory=torch.cuda.max_memory_allocated()/1024.0/1024.0))
            # write to tensorboardX
            loss_global_avg = meters.loss.global_avg
            loss_median = meters.loss.median
            # print('loss global average: {0}, loss median: {1}'.format(meters.loss.global_avg, meters.loss.median))
            summary_writer.add_scalar('train_loss_global_avg', loss_global_avg, iteration)
            summary_writer.add_scalar('train_loss_median', loss_median, iteration)
            summary_writer.add_scalar('train_loss_raw', losses_reduced, iteration)
            # summary_writer.add_scalar('distillation_losses_raw', distillation_losses, iteration)
            summary_writer.add_scalar('faster_rcnn_losses_raw', faster_rcnn_losses, iteration)
            # summary_writer.add_scalar('distillation_losses_avg', average_distillation_loss, iteration)
            summary_writer.add_scalar('faster_rcnn_losses_avg', average_faster_rcnn_loss, iteration)
        # Every time meets the checkpoint_period, save the target model (parameters)
        if iteration % checkpoint_period == 0:
            checkpointer_target.save("model_{:07d}".format(iteration), **arguments_target)
        # When meets the last iteration, save the target model (parameters)
        if iteration == max_iter:
            checkpointer_target.save("model_final", **arguments_target)
    if args.stat:
        pred_cat_cnt = Counter(pred_catlist)
        # print(pred_cat_cnt)
        from maskrcnn_benchmark.data.datasets import voc
        print("Pred Instance Class Statistics")
        old_classes = list(voc.CLS_SETS[cfg_target.MODEL.ROI_BOX_HEAD.SPLIT][0])
        tpt = "{}:\t {}\t"
        for x in old_classes:
            num = pred_cat_cnt[x] if x in pred_cat_cnt.keys() else 0
            print(tpt.format(x, num))
        from ipdb import set_trace; set_trace()
        
    # Display the total used training time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time/max_iter))


def train(cfg_source, logger_source, cfg_target, logger_target, distributed):

    model_source = build_detection_model(cfg_source)  # create the source model
    model_target = build_detection_model(cfg_target)  # create the target model
    device = torch.device(cfg_source.MODEL.DEVICE)  # default is "cuda"
    model_target.to(device)  # move target model to gpu
    model_source.to(device)  # move source model to gpu
    optimizer = make_optimizer(cfg_target, model_target)  # config optimization strategy
    scheduler = make_lr_scheduler(cfg_target, optimizer)  # config learning rate
    # initialize mixed-precision training
    use_mixed_precision = cfg_target.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model_target, optimizer = amp.initialize(model_target, optimizer, opt_level=amp_opt_level)
    # create a parameter dictionary and initialize the iteration number to 0
    arguments_target = {}
    arguments_target["iteration"] = 0
    arguments_source = {}
    arguments_source["iteration"] = 0
    # path to store the trained parameter value
    output_dir_target = cfg_target.OUTPUT_DIR
    output_dir_source = cfg_source.OUTPUT_DIR
    # create summary writer for tensorboard
    summary_writer = SummaryWriter(log_dir=cfg_target.TENSORBOARD_DIR)
    # when only use 1 gpu, get_rank() returns 0
    save_to_disk = get_rank() == 0
    # create check pointer for source model & load the pre-trained model parameter to source model
    checkpointer_source = DetectronCheckpointer(cfg_source, model_source, optimizer=None, scheduler=None, save_dir=output_dir_source,
                                                save_to_disk=save_to_disk, logger=logger_source)
    extra_checkpoint_data_source = checkpointer_source.load(cfg_source.MODEL.WEIGHT)
    # create check pointer for target model & load the pre-trained model parameter to target model
    checkpointer_target = DetectronCheckpointer(cfg_target, model_target, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir_target,
                                                save_to_disk=save_to_disk, logger=logger_target)
    extra_checkpoint_data_target = checkpointer_target.load(cfg_target.MODEL.WEIGHT)
    # dict updating method to update the parameter dictionary for source model
    arguments_source.update(extra_checkpoint_data_source)
    # dict updating method to update the parameter dictionary for target model
    arguments_target.update(extra_checkpoint_data_target)
    print('start iteration: {0}'.format(arguments_target["iteration"]))
    # load training data
    data_loader = make_data_loader(cfg_target, is_train=True, is_distributed=distributed, start_iter=arguments_target["iteration"])
    print('finish loading data')
    # number of iteration to store parameter value in pth file
    checkpoint_period = cfg_target.SOLVER.CHECKPOINT_PERIOD

    # train the model using overwrite function
    do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_source, checkpointer_target,
             device, checkpoint_period, arguments_source, arguments_target, summary_writer, cfg_target)

    return model_target


def test(cfg_target, model, distributed):

    if distributed:  # whether use multiple gpu to train
        model = model.module
    # Release unoccupied memory
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg_target.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg_target.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    # according to number of test dataset decides number of output folder
    output_folders = [None] * len(cfg_target.DATASETS.TEST)
    # create folder to store test result
    # output result is stored in: cfg.OUTPUT_DIR/inference/cfg.DATASETS.TEST
    dataset_names = cfg_target.DATASETS.TEST
    if cfg_target.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg_target.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    # load testing data
    print('loading test data')
    data_loaders_val = make_data_loader(cfg_target, is_train=False, is_distributed=distributed)
    print('finish loading test data')
    # test
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(model,
                  data_loader_val,
                  dataset_name=dataset_name,
                  iou_types=iou_types,
                  box_only=False if cfg_target.MODEL.RETINANET_ON else cfg_target.MODEL.RPN_ONLY,
                  device=cfg_target.MODEL.DEVICE,
                  expected_results=cfg_target.TEST.EXPECTED_RESULTS,
                  expected_results_sigma_tol=cfg_target.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                  output_folder=output_folder,
                  alphabetical_order=cfg_target.TEST.COCO_ALPHABETICAL_ORDER)
        # synchronize function for multiple gpu inference
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--src-file",
        default="",
        metavar="SRC_FILE",
        help="path to src config file",
        type=str,
    )
    parser.add_argument(
        "--tat-file",
        default="",
        metavar="TAT_FILE",
        help="path to target config file",
        type=str,
    )
    parser.add_argument(
        "--vis",
        help="visualize mode",
        action="store_true",
    )
    parser.add_argument(
        "--stat",
        help="statistic mode",
        action="store_true",
    )
    global args
    args = parser.parse_args()
    # source_model_config_file = "/home/zhengkai/Faster-ILOD/configs/e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml"
    # target_model_config_file = "/home/zhengkai/Faster-ILOD/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml"
    source_model_config_file = args.src_file
    target_model_config_file = args.tat_file
    # source_model_config_file = "/home/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml"
    # target_model_config_file = "/home/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml"
    local_rank = 0

    # get the number of gpu from the device
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed: whether using multiple gpus to train
    distributed = num_gpus > 1

    YML_ROOT = '/data1/pbdata/data_zk/Faster-ILOD'
    cfg_source = cfg.clone()
    cfg_source.merge_from_file(source_model_config_file)
    subset = source_model_config_file.split('/')[2]
    yaml_name = source_model_config_file.split('/')[3]
    if cfg_source.MODEL.WEIGHT == "": # if none, then make up for it automatically
        cfg_source.MODEL.WEIGHT = os.path.join(YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "first_step", "model_final.pth")
    cfg_source.OUTPUT_DIR = os.path.join(YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "source")
    cfg_source.TENSORBOARD_DIR = os.path.join(YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "source", "tensorboard")
    cfg_source.freeze()
    cfg_target = cfg.clone()
    cfg_target.merge_from_file(target_model_config_file)
    subset = target_model_config_file.split('/')[2]
    yaml_name = target_model_config_file.split('/')[3]
    if cfg_target.MODEL.WEIGHT == "": # if none, then make up for it automatically
        cfg_target.MODEL.WEIGHT = os.path.join(YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "first_step", "model_trim_optimizer_iteration.pth")
    cfg_target.OUTPUT_DIR = os.path.join(YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "target")
    cfg_target.TENSORBOARD_DIR = os.path.join(YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "target", "tensorboard")
    # only run through one epoch
    if args.stat:
        cfg_target.SOLVER.MAX_ITER = None
    cfg_target.freeze()

    output_dir_target = cfg_target.OUTPUT_DIR
    if output_dir_target:
        mkdir(output_dir_target)
    output_dir_source = cfg_source.OUTPUT_DIR
    if output_dir_source:
        mkdir(output_dir_source)
    tensorboard_dir = cfg_target.TENSORBOARD_DIR
    if tensorboard_dir:
        mkdir(tensorboard_dir)

    logger_target = setup_logger("maskrcnn_benchmark_target_model", output_dir_target, get_rank())
    logger_target.info("config yaml file for target model: {}".format(target_model_config_file))
    logger_target.info("local rank: {}".format(local_rank))
    logger_target.info("Using {} GPUs".format(num_gpus))
    logger_target.info("Collecting env info (might take some time)")
    logger_target.info("\n" + collect_env_info())
    # open and read the input yaml file, store it on source config_str and display on the screen
    with open(target_model_config_file, "r") as cf:
        target_config_str = "\n" + cf.read()
        logger_target.info(target_config_str)
    logger_target.info("Running with config:\n{}".format(cfg_target))

    logger_source = setup_logger("maskrcnn_benchmark_source_model", output_dir_source, get_rank())
    logger_source.info("config yaml file for target model: {}".format(source_model_config_file))
    logger_source.info("local rank: {}".format(local_rank))
    logger_source.info("Using {} GPUs".format(num_gpus))
    logger_source.info("Collecting env info (might take some time)")
    logger_source.info("\n" + collect_env_info())
    # open and read the input yaml file, store it on source config_str and display on the screen
    with open(source_model_config_file, "r") as cf:
        source_config_str = "\n" + cf.read()
        logger_source.info(source_config_str)
    logger_source.info("Running with config:\n{}".format(cfg_source))

    # start to train the model
    model_target = train(cfg_source, logger_source, cfg_target, logger_target, distributed)
    # start to test the trained target model
    test(cfg_target, model_target, distributed)


def overlay_boxes(image, predictions, color):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    # colors = self.compute_colors_for_labels(labels).tolist()

    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        # set_trace()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), color, 1
        )

    return image

def get_groundtruth(img_name, annopath, old_classes, new_classes, exclude_classes=[]):
    anno = ET.parse(annopath % img_name).getroot()

    def _preprocess_annotation(target, old_classes, new_classes, exclude_classes=[]):
        cls = old_classes + new_classes
        class_to_ind = dict(zip(cls, range(len(cls))))
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if difficult:
                continue
            name = obj.find("name").text.lower().strip()
            # 如果该图片中的gt box是old class/exclude class，则不添加到gt box的list中
            old_class_flag = False
            for old in old_classes:
                if name == old:
                    old_class_flag = True
                    break
            exclude_class_flag = False
            for exclude in exclude_classes:
                if name == exclude:
                    exclude_class_flag = True
                    break

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]
            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            boxes.append(bndbox)
            gt_classes.append(class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    anno = _preprocess_annotation(anno, old_classes, new_classes, exclude_classes)
    height, width = anno["im_info"]
    target = BoxList(anno["boxes"], (width, height), mode="xyxy")
    target.add_field("labels", anno["labels"])
    target.add_field("difficult", anno["difficult"])
    return target


if __name__ == "__main__":
    main()

