import os

import torch
import torch.utils.data
from PIL import Image
import sys
import scipy.io as scio
import cv2
import numpy

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList
import copy

# {split: (old_class, new_class)}
CLS_SETS = {
    '10+10': (("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow"),
            ("diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")),
    '15+5': (("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
            "diningtable", "dog", "horse", "motorbike", "pottedplant"),
            ("sheep", "sofa", "train", "tvmonitor", "person")),
    '19+1': (("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "pottedplant", "sheep", "sofa", "train", "tvmonitor"),
            ("person",))
}
# 注意19+1，1的tuple[1]需要额外加一个, 否则会将字符串拆分

class PascalVOCDataset(torch.utils.data.Dataset):
    # 第二阶段，则用全数据集
    # CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    #            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    第一阶段
    如果挑选出来的new class有在old class之前的，则需要手动调整CLASSES，去掉new class
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    # def __init__(self, data_dir, split, use_difficult=False, transforms=None, external_proposal=False, old_classes=[],
    #              new_classes=[], excluded_classes=[], is_train=True):
    def __init__(self, data_dir, split, use_difficult=False, transforms=None, external_proposal=False, cfg=None, is_train=True):
        self.root = data_dir
        self.image_set = split  # train, validation, test
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.use_external_proposal = external_proposal

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self._proposalpath = os.path.join(self.root, "EdgeBoxesProposals", "%s.mat")

        self._img_height = 0
        self._img_width = 0

        # self.old_classes = old_classes
        # self.new_classes = new_classes
        # self.exclude_classes = excluded_classes
        self.is_train = is_train
        self.cfg = cfg
        self.is_incremental = cfg.MODEL.ROI_BOX_HEAD.INCREMENTAL
        if not self.is_incremental:
            self.old_classes = []
            self.new_classes = list(CLS_SETS[cfg.MODEL.ROI_BOX_HEAD.SPLIT][0])
            self.exclude_classes = list(CLS_SETS[cfg.MODEL.ROI_BOX_HEAD.SPLIT][1])
            new_copy = copy.deepcopy(self.new_classes)
            new_copy.insert(0, "__background__ ")
            self.CLASSES = tuple(new_copy)
        else:
            self.old_classes = list(CLS_SETS[cfg.MODEL.ROI_BOX_HEAD.SPLIT][0])
            self.new_classes = list(CLS_SETS[cfg.MODEL.ROI_BOX_HEAD.SPLIT][1])
            self.exclude_classes = []
            new_copy = copy.deepcopy(self.old_classes+self.new_classes)
            new_copy.insert(0, "__background__ ")
            self.CLASSES = tuple(new_copy)
        # from ipdb import set_trace; set_trace()
        # load data from all categories
        # self._normally_load_voc()

        # do not use old data
        if self.is_train:  # training mode
            print('voc.py | in training mode')
            if self.cfg.MODEL.DATASET_FITER:
                print('voc.py | filter data with old instance')
                self._load_img_from_NEW_cls_without_old_cls()
            else:
                print('voc.py | keep data with old instance')
                self._load_img_from_NEW_cls_without_old_data()
        else:
            print('voc.py | in test mode')
            self._load_img_from_NEW_and_OLD_cls_without_old_data()

    def _normally_load_voc(self):
        # not use
        """ load data from all 20 categories """

        print("voc.py | normally_load_voc | load data from all 20 categories")
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.final_ids = self.ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # image_index : image_id

        # cls = PascalVOCDataset.CLASSES
        cls = tuple(CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][0] \
            + CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][1] \
            + "__background__ ")
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class_name : class_id

    def _load_img_from_NEW_and_OLD_cls_without_old_data(self):
        """
        测试的时候，new + old
        """
        self.ids = []
        total_classes = self.new_classes + self.old_classes
        for w in range(len(total_classes)):
            category = total_classes[w]
            img_per_categories = []
            with open(self._imgsetpath % "{0}_{1}".format(category, self.image_set)) as f:
                buff = f.readlines()
            buff = [x.strip("\n") for x in buff]
            for i in range(len(buff)):
                a = buff[i]
                b = a.split(' ')
                if b[1] == "-1":  # do not contain the category object
                    pass
                elif b[2] == '0':  # contain the category object -- difficult level
                    if self.is_train:
                        pass
                    else:
                        img_per_categories.append(b[0])
                        self.ids.append(b[0])
                else:
                    img_per_categories.append(b[0])
                    self.ids.append(b[0])
            print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | number of images in {0}_{1}: {2}'.format(category, self.image_set, len(img_per_categories)))

        # check for image ids repeating
        self.final_ids = []
        for id in self.ids:
            repeat_flag = False
            for final_id in self.final_ids:
                if id == final_id:
                    repeat_flag = True
                    break
            if not repeat_flag:
                self.final_ids.append(id)
        print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = self.CLASSES # PascalVOCDataset.CLASSES
        # cls = CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][0] + CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][1]
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        # from ipdb import set_trace; set_trace()
    def _load_img_from_NEW_cls_without_old_data(self):
        """
        根据yml中指定的new class data按照类别从class_train.txt中取，把图片id放到img_ids_per_category和ids中
        """
        self.ids = []
        for incremental in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f: # ./voc/VOC2007/ImageSets/Main/cat_train.txt
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff] # image list contains this category

            for i in range(len(buff)):
                x = buff[i]
                x = x.split(' ')
                if x[1] == '-1':
                    pass
                elif x[2] == '0':  # include difficult level object
                    if self.is_train:
                        pass
                    else: # 测试的时候才用difficult level的object
                        img_ids_per_category.append(x[0])
                        self.ids.append(x[0])
                else:
                    img_ids_per_category.append(x[0])
                    self.ids.append(x[0])
            print('voc.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            # 由于是按照类别取的img id（一张图中可能存在两个object），会被多个class loop重复取，所以要去重
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)
            print('voc.py | load_img_from_NEW_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = self.CLASSES # PascalVOCDataset.CLASSES
        # if self.is_incremental:
        #     cls = CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][0] + CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][1]
        # else:
        #     cls = self.old_classes
        # from ipdb import set_trace; set_trace()
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def _load_img_from_NEW_cls_without_old_cls(self):
        """
        根据yml中指定的new class data按照类别从class_train.txt中取，把图片id放到img_ids_per_category和ids中
        + 仅载入含有new category的img
        """
        self.ids = []
        for incremental in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f: # ./voc/VOC2007/ImageSets/Main/cat_train.txt
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff] # image list contains this category

            for i in range(len(buff)):
                x = buff[i]
                x = x.split(' ')
                if x[1] == '-1':
                    pass
                elif x[2] == '0':  # include difficult level object
                    if self.is_train:
                        pass
                    else: # 测试的时候才用difficult level的object
                        img_ids_per_category.append(x[0])
                        self.ids.append(x[0])
                else:
                    img_ids_per_category.append(x[0])
                    self.ids.append(x[0])
            # print('voc.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            # 由于是按照类别取的img id（一张图中可能存在两个object），会被多个class loop重复取，所以要去重
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)
            # print('voc.py | load_img_from_NEW_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        print('filter before | load_img_from_NEW_cls_without_old_cls | total used number of images in {0}: {1}'.format(
            self.image_set, len(self.final_ids)))


        no_old_cls_final_ids = []

        # @zhengkai filter image has old instances
        for i, img_id in enumerate(self.final_ids):
            old_class_flag = False
            anno = ET.parse(self._annopath % img_id).getroot()
            for obj in anno.iter("object"):
                difficult = int(obj.find("difficult").text) == 1
                if not self.keep_difficult and difficult:
                    continue
                name = obj.find("name").text.lower().strip()
                for old in self.old_classes:
                    if name == old:
                        old_class_flag = True
                        break

                if old_class_flag:
                    break
            
            if not old_class_flag:
                no_old_cls_final_ids.append(img_id)
        
        self.final_ids = no_old_cls_final_ids
        
        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        print('filter after | load_img_from_NEW_cls_without_old_cls | total used number of images in {0}: {1}'.format(
            self.image_set, len(self.final_ids)))

            
        cls = self.CLASSES # PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))


    def __getitem__(self, index):
        img_id = self.final_ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.use_external_proposal: # F
            proposal = self.get_proposal(index)
            proposal = proposal.clip_to_image(remove_empty=True)
        else:
            proposal = None

        # draw_image(img, target, proposal, "{0}_{1}_voc_getitem".format(index, img_id))
        restore_list = []
        if self.transforms is not None:
            if self.cfg.MODEL.MULTI_TEACHER:
                img, target, proposal, restore_list = self.transforms(img, target, proposal)
            else:
                img, target, proposal = self.transforms(img, target, proposal)

        if self.cfg.MODEL.MULTI_TEACHER:
            return img, target, proposal, index, restore_list
        return img, target, proposal, index

    def __len__(self):
        return len(self.final_ids)

    def get_groundtruth(self, index):
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno) # {'boxes': tensor, "labels": tensor, "difficult": tensor, "im_info":xx}

        height, width = anno["im_info"]
        self._img_height = height
        self._img_width = width
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def get_proposal(self, index):
        boxes = []

        img_id = self.final_ids[index]
        proposal_path = self._proposalpath % "{0}".format(img_id)
        proposal_raw_data = scio.loadmat(proposal_path)
        proposal_data = proposal_raw_data['bbs']
        proposal_length = proposal_data.shape[0]
        for i in range(2000):
            # print('i: {0}'.format(i))
            if i >= proposal_length:
                break
            left = proposal_data[i][0]
            top = proposal_data[i][1]
            width = proposal_data[i][2]
            height = proposal_data[i][3]
            score = proposal_data[i][4]
            right = left + width
            bottom = top + height
            box = [left, top, right, bottom]
            boxes.append(box)
        img_height = self._img_height
        img_width = self._img_width

        boxes = torch.tensor(boxes, dtype=torch.float32)
        proposal = BoxList(boxes, (img_width, img_height), mode="xyxy")

        return proposal

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            # 如果该图片中的gt box是old class/exclude class，则不添加到gt box的list中
            old_class_flag = False
            for old in self.old_classes:
                if name == old:
                    old_class_flag = True
                    break
            exclude_class_flag = False
            for exclude in self.exclude_classes:
                if name == exclude:
                    exclude_class_flag = True
                    break

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]
            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            if exclude_class_flag:
                # print('voc.py | incremental train | object category belongs to exclude categoires: {0}'.format(name))
                pass
            elif self.is_train and old_class_flag:
                # print('voc.py | incremental train | object category belongs to old categoires: {0}'.format(name))
                pass
            else:
                boxes.append(bndbox)
                gt_classes.append(self.class_to_ind[name])
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

    def get_img_info(self, index):
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return self.CLASSES[class_id] # PascalVOCDataset.CLASSES[class_id]
        # if self.is_incremental:
        #     cls = CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][0] + CLS_SETS[self.cfg.MODEL.ROI_BOX_HEAD.SPLIT][1]
        # else:
        #     cls = self.old_classes
        # return cls[class_id]

    def get_img_id(self, index):
        img_id = self.final_ids[index]
        return img_id


def main():
    data_dir = "/home/DATA/VOC2007"
    split = "test"  # train, val, test
    use_difficult = False
    transforms = None
    dataset = PascalVOCDataset(data_dir, split, use_difficult, transforms)


if __name__ == '__main__':
    main()
