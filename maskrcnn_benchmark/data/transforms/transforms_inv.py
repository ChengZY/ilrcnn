# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# add original method with output of transform or not
# add reverse method
import random
import copy
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        # self.index = 0

    def __call__(self, image, target, proposal=None):
        restore_list = []
        for t in self.transforms:
            # draw_image(image, target, proposal, self.index)
            # self.index = self.index + 1
            image, target, proposal, restore_params = t(image, target, proposal)
            restore_list.append(restore_params)
        return image, target, proposal, restore_list

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        # self.index = 0

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
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

    def __call__(self, image, target, proposal):
        sz_origin = copy.copy(image.size)
        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is not None:
            target = target.resize(image.size)
        if proposal is not None:
            proposal = proposal.resize(image.size)
        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1

        return image, target, proposal, sz_origin


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        # self.index = 0

    def __call__(self, image, target, proposal):
        trans_used = False
        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)
            if proposal is not None:
                proposal = proposal.transpose(0)
        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1
            trans_used = True
        return image, target, proposal, trans_used


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target, proposal):
        image = self.color_jitter(image)
        return image, target, proposal, None


class ToTensor(object):
    def __call__(self, image, target, proposal):
        return F.to_tensor(image), target, proposal, None


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target, proposal):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, proposal, None


####### Inverse Transform Method #######
class Resize_Reverse(object):
    def __call__(self, target, origin_sz):
        target = target.resize(origin_sz)
        return target  


class RandomHorizontalFlip_Reverse(object):
    def __call__(self, target, hor_fliped):
        if hor_fliped:
            target = target.transpose(FLIP_LEFT_RIGHT)
            # maybe used for retinaNet based method, do not need 'sel_ind'
            # target.extra_fields['sel_ind'] = torch.tensor(index_transpose(target.extra_fields['sel_ind'].cpu(),target.get_field('feat_w_h'),dim=FLIP_LEFT_RIGHT))
        return target


class ColorJitter_Reverse(object):
     def __call__(self,target,par):
        return target


class ToTensor_Reverse(object):
     def __call__(self,target,par):
            return target


class Normalize_Reverse(object):
     def __call__(self,target,par):
            return target


#./maskrcnn_benchmark/data/transforms/build.py:40
def trans_reverse(target, reverse_info):
    target_out = copy.deepcopy(target)
    if target.bbox.shape[0] == 0:
        target_out.size = reverse_info[1]
        return target_out
    transform = [
            ColorJitter_Reverse(),
            Resize_Reverse(),
            RandomHorizontalFlip_Reverse(),
            ToTensor_Reverse(),
            Normalize_Reverse(),
        ]
    for _fn, _para in zip(transform, reverse_info):
        target_out = _fn(target_out,_para)
    return target_out
    
