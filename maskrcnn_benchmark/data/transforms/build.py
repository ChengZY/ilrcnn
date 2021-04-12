# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from . import transforms_inv as T_I

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    if cfg.MODEL.MULTI_TEACHER:
        TRANS = T_I
    else:
        TRANS = T

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = TRANS.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255)
    color_jitter = TRANS.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    transform = TRANS.Compose(
        [
            color_jitter,  # modify brightness, contrast & saturation
            TRANS.Resize(min_size, max_size),
            TRANS.RandomHorizontalFlip(flip_prob),  # horizontally flip the image in percentage of 0.5
            TRANS.ToTensor(),  # convert to tensor
            normalize_transform,
        ]
    )
    return transform
