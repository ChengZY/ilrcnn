# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
# from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size, is_main_process
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.inference import _accumulate_predictions_from_multiple_gpus, compute_on_dataset
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.data.datasets.evaluation.voc.voc_eval import eval_detection_voc
from maskrcnn_benchmark.data.datasets.voc import CLS_SETS
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
from tqdm import tqdm
import numpy as np
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
# CATEGORIES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    YML_ROOT = '/data1/pbdata/data_zk/Faster-ILOD'
    cfg_file = args.config_file
    subset = cfg_file.split('/')[2]
    yaml_name = cfg_file.split('/')[3]
    cfg.OUTPUT_DIR = os.path.join(
        YML_ROOT + "/incremental_learning_ResNet50_C4/", subset, "target")
    # overwrite the load checkpoint
    cfg.MODEL.WEIGHT = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    # print("= " * 50)
    # print(cfg.MODEL.WEIGHT)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        pred_list, gt_list = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            alphabetical_order=cfg.TEST.COCO_ALPHABETICAL_ORDER
        )
        synchronize()
    from ipdb import set_trace; set_trace()
    # 为方便可视化一些misclassify，去掉对角线上的value
    pred_list = np.array(pred_list)
    gt_list = np.array(gt_list)
    dis_mask = (gt_list != pred_list)
    gt_list = gt_list[dis_mask]
    pred_list = pred_list[dis_mask]
    fig_path = './confusion_{}.png'.format(subset)
    C = confusion_matrix(gt_list, pred_list, labels=np.arange(21).tolist())
    CATEGORIES = list(CLS_SETS[cfg.MODEL.ROI_BOX_HEAD.SPLIT][0] + CLS_SETS[cfg.MODEL.ROI_BOX_HEAD.SPLIT][1])
    CATEGORIES.insert(0, "background")
    df = pd.DataFrame(C, index=CATEGORIES, columns=CATEGORIES)
    fig = sns.heatmap(df)

    hp = fig.get_figure()
    hp.savefig(fig_path, dpi=400)

# overwrite inference to do confusion matrix in voc_evaluation
def inference(model, data_loader, dataset_name, iou_types=("bbox",), box_only=False, device="cuda",
        expected_results=(), expected_results_sigma_tol=4, output_folder=None, external_proposal=False, alphabetical_order=False):

    print('inference.py | alphabetical_order: {0}'.format(alphabetical_order))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    # logger = logging.getLogger("maskrcnn_benchmark_target_model.inference")
    dataset = data_loader.dataset
    # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer, external_proposal)
    # from ipdb import set_trace; set_trace()
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    # logger.info(
    #     "Total run time: {} ({} s / img per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )
    total_infer_time = get_time_str(inference_timer.total_time)
    # logger.info(
        # "Model inference time: {} ({} s / img per device, on {} devices)".format(
            # total_infer_time,
            # inference_timer.total_time * num_devices / len(dataset),
            # num_devices,
        # )
    # )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    print('inference.py | alphabetical_order: {0}'.format(alphabetical_order))
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        alphabetical_order=alphabetical_order
    )

    # return evaluate(dataset=dataset,
    #                 predictions=predictions,
    #                 output_folder=output_folder,
    #                 **extra_args)

    return do_confusion_matrix(dataset, predictions, output_folder)

def do_confusion_matrix(dataset, predictions, output_dir):
    # pred_boxlists = []
    # gt_boxlists = []
    pred_list = []
    gt_list = []
    cmatrix = np.zeros((21,21), dtype=np.uint32)
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        # pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        # gt_boxlists.append(gt_boxlist)

        """
        每张图片的gt和pred操作，计算iou矩阵，根据iou给每个pred分配gt，看label是否匹配
        """
        pred_box = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()
        # filter out difficult box
        valid_mask = gt_difficult == 0
        gt_bbox = gt_bbox[valid_mask]
        gt_label = gt_label[valid_mask]
        gt_difficult  = gt_difficult[valid_mask]

        iou = boxlist_iou(
            BoxList(pred_box, prediction.size),
            BoxList(gt_bbox, gt_boxlist.size),
        ).numpy()
        gt_index = iou.argmax(axis=1)
        IOU_THRESH = 0.5
        over_thresh_mask = iou.max(axis=1) >= IOU_THRESH
        gt_index[iou.max(axis=1) < IOU_THRESH] = -1 # (nr_pred, )
        pred_label_n = pred_label[over_thresh_mask]
        # filter -1 gt_index
        valid_gt_index = np.array(list(filter(lambda x:x!=-1, gt_index))).astype(np.uint8)
        gt_label_n = gt_label[valid_gt_index]
        
        assert len(pred_label_n) == len(gt_label_n)
        # for row, col in zip(pred_label_n, gt_label_n):
        #     cmatrix[row, col] += 1
        # from ipdb import set_trace; set_trace()
        pred_list.extend(pred_label_n.tolist())
        gt_list.extend(gt_label_n.tolist())
    # result = eval_detection_voc(
    #     pred_boxlists=pred_boxlists,
    #     gt_boxlists=gt_boxlists,
    #     iou_thresh=0.5,
    #     use_07_metric=True,
    # )
    # result_str = "mAP: {:.4f}\n".format(result["map"])
    # for i, ap in enumerate(result["ap"]):
    #     if i == 0:  # skip background
    #         continue
    #     result_str += "{:<16}: {:.4f}\n".format(
    #         dataset.map_class_id_to_class_name(i), ap
    #     )
    # logger.info(result_str)
    # if output_folder:
    #     with open(os.path.join(output_folder, "result.txt"), "w") as fid:
    #         fid.write(result_str)
    return pred_list, gt_list


if __name__ == "__main__":
    main()
