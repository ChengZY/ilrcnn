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
from torchvision import transforms as T
from torchvision.transforms import functional as F

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
from maskrcnn_benchmark.structures.image_list import to_image_list
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
from collections import Counter
from ipdb import set_trace
warnings.filterwarnings("ignore", category=UserWarning)


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
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
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

    cfg_source = cfg.clone()
    cfg_source.merge_from_file(source_model_config_file)
    subset = source_model_config_file.split('/')[2]
    yaml_name = source_model_config_file.split('/')[3]
    cfg_source.freeze()
    cfg_target = cfg.clone()
    cfg_target.merge_from_file(target_model_config_file)
    subset = target_model_config_file.split('/')[2]
    yaml_name = target_model_config_file.split('/')[3]
    cfg_target.freeze()

    model_source = build_detection_model(cfg_source)
    model_target = build_detection_model(cfg_target)
    device = torch.device(cfg_source.MODEL.DEVICE)  # default is "cuda"
    model_target.to(device)  # move target model to gpu
    model_source.to(device)  # move source model to gpu

    ckpt_source_path = "/home/zhengkai/Faster-ILOD/incremental_learning_ResNet50_C4/10_10/source/model_final.pth"
    ckpt_target_path = "/home/zhengkai/Faster-ILOD/incremental_learning_ResNet50_C4/10_10/model_final.pth"
    ckpt_source = torch.load(ckpt_source_path, map_location='cpu')
    ckpt_target = torch.load(ckpt_target_path, map_location='cpu')

    model_source.load_state_dict(ckpt_source['model'])
    model_target.load_state_dict(ckpt_target['model'])

    
    # output_folders = [None] * len(cfg_target.DATASETS.TEST)
    # create folder to store test result
    # output result is stored in: cfg.OUTPUT_DIR/inference/cfg.DATASETS.TEST
    dataset_names = cfg_target.DATASETS.TEST # 'voc_2007_test'

    voc_src = VOCDemo(
        cfg=cfg_source,
        confidence_threshold=args.confidence_threshold,
        min_image_size=args.min_image_size,
        model=model_source,
        is_old=True,
    )

    voc_tat = VOCDemo(
        cfg=cfg_target,
        confidence_threshold=args.confidence_threshold,
        min_image_size=args.min_image_size,
        model=model_target,
        is_old=False,
    )

    # load 10_10 dataset
    # old_class = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    #             "bus", "car", "cat", "chair", "cow"]
    # new_class = ["diningtable", "dog", "horse", "motorbike", "person",
    #             "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # load 15_5 dataset
    old_class = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable", 
                "dog", "horse", "motorbike", "pottedplant"]
    new_class = ["sheep", "sofa", "train", "tvmonitor", "person"]
    # load 19_1 dataset
    # old_class = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    #             "bus", "car", "cat", "chair", "cow", "diningtable", 
    #             "dog", "horse", "motorbike", "pottedplant", 
    #             "sheep", "sofa", "train", "tvmonitor"]
    # new_class = ["person",]

    ROOT = "/home/zhengkai/maskrcnn-benchmark/datasets/voc/VOC2007"
    annopath = os.path.join(ROOT, "Annotations", "%s.xml")
    imgpath = os.path.join(ROOT, "JPEGImages", "%s.jpg")
    imgsetpath = os.path.join(ROOT, "ImageSets", "Main", "%s.txt")


    """
    统计10_10的数据分类中，后者包含前者的class的instance数量（即被missing掉的old ann个数）
    """
    image_set = 'train'
    stat_groundtruth(new_class, old_class, imgsetpath, image_set, annopath)


    """
    # 可视化new model和old model在old class上的表现
    category = "cow"
    image_set = "val"

    # load one category per time
    with open(imgsetpath % "{}_{}".format(category, image_set)) as f:
        buff = f.readlines()
    img_per_categories = []
    buff = [x.strip("\n") for x in buff]
    for i in range(len(buff)):
        a = buff[i]
        b = a.split(' ')
        if b[1] == '-1': pass # not contain this category object
        elif b[2] == '0': img_per_categories.append(b[0]) # if difficult level
        else: img_per_categories.append(b[0])
    
    # load image here
    for img_name in img_per_categories:
        img = cv2.imread(imgpath % img_name)
        print(img.shape)
        
        target = get_groundtruth(img_name, annopath, old_class, new_class)
        # set_trace()
        target = target.clip_to_image(remove_empty=True)
        image_copy, src_toppred = voc_src(img)
        _, tat_toppred = voc_tat(img)
        # show the result on one image
        composite = voc_src.overlay_boxes(image_copy, src_toppred)
        composite = voc_src.overlay_class_names(composite, src_toppred)
        composite = voc_tat.overlay_boxes(composite, tat_toppred)
        composite = voc_tat.overlay_class_names(composite, tat_toppred)
        cv2.imshow("VOC", composite)
        if cv2.waitKey(0) == 27: # 'ESC'
            cv2.destroyAllWindows() 
            exit(0)
    """

def stat_groundtruth(new_classes, old_classes, imgsetpath, image_set, annopath):
    """
    取new class的image，把image加入到list中，如果存在就不再次stat；否则：
    - 看该image中old instance的个数
    """
    obj_classes = new_classes + old_classes
    train_img = []
    has_old_img = []
    inst = []
    has_old = 0
    obj_cnt = 0
    total_old_cnt = 0
    max_old_cnt_perimg = 0
    min_old_cnt_perimg = 999999
    for x in new_classes:
        print("Processing class: ", x)
        with open(imgsetpath % "{}_{}".format(x, image_set)) as f:
            buff = f.readlines()
        # img_per_categories = []
        buff = [x.strip("\n") for x in buff]
        # 对每张图片
        for i in range(len(buff)):
            has_old_flag = False
            a = buff[i]
            b = a.split(' ')

            if b[0] in train_img: # 如果已经统计过了，则跳过
                continue

            if b[1] == '-1': continue # not contain this category object
            elif b[1] == '0': train_img.append(b[0]) # if difficult level
            else: train_img.append(b[0])

            ## 统计已经存在的new data中old instance的数量
            anno = ET.parse(annopath % b[0]).getroot()
            # 遍历每个object
            old_cnt = 0
            for obj in anno.iter("object"):
                difficult = int(obj.find("difficult").text) == 1
                if difficult: # 若为difficult则不统计该instance
                    continue
                name = obj.find("name").text.lower().strip()
                if name in old_classes:
                    old_cnt += 1
                    has_old_flag = True
                if name in obj_classes:
                    obj_cnt += 1
                inst.append(name)
            total_old_cnt += old_cnt
            max_old_cnt_perimg = max(max_old_cnt_perimg, old_cnt)
            min_old_cnt_perimg = min(min_old_cnt_perimg, old_cnt)
            
            if has_old_flag:
                if not (b[0] in has_old_img):
                    has_old_img.append(b[0])
        # from ipdb import set_trace; set_trace()
    print('*' * 30)
    print("training image number: ", len(train_img))
    print("training image number(has old instance): ", len(has_old_img))
    print("old instance/ total instance: {}/{}".format(total_old_cnt, obj_cnt))
    print("max/min old instance number for one image: {}/{}".format(max_old_cnt_perimg, min_old_cnt_perimg))

    inst_cnt = Counter(inst)      
    print("Instance Statistics")
    # from ipdb import set_trace; set_trace()
    # print(inst_cnt)
    tpt = "{}:\t {}\t"
    print("======= OLD SET =========")
    for x in old_classes:
        num = inst_cnt[x] if x in inst_cnt.keys() else 0
        print(tpt.format(x, num))
    print("======= NEW SET =========")
    for x in new_classes:
        num = inst_cnt[x] if x in inst_cnt.keys() else 0
        print(tpt.format(x, num))


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



class VOCDemo(object):

    CATEGORIES = ["__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    def __init__(self, cfg, min_image_size, model, confidence_threshold=0.7, is_old=False):
        self.cfg = cfg.clone()
        self.min_image_size = min_image_size
        self.model = model
        self.model.eval()
        self.transforms = self.build_transform()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.confidence_threshold = confidence_threshold
        self.cpu_device = torch.device("cpu")
        self.is_old = is_old
        self.color = (0,205,0) if self.is_old else (250,0,0) # 如果是old model, green
        self.thickness = 2 # 2 if self.is_old else 1

    def build_transform(self):
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
    
    def __call__(self, image):
        """
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        #result = self.overlay_boxes(result, top_predictions)
        # result = self.overlay_class_names(result, top_predictions)

        return result, top_predictions

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)[0] # 2-len tuple returned
        # set_trace()
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def overlay_boxes(self, image, predictions):
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
                image, tuple(top_left), tuple(bottom_right), tuple(self.color), 1
            )

        return image

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2] if self.is_old else box[2:4]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, self.color, self.thickness
            )

        return image

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

if __name__ == "__main__":
    main()

