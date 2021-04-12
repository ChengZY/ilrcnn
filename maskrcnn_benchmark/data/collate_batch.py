# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0, cfg=None):
        self.size_divisible = size_divisible
        self.cfg = cfg

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))  # data get from Dataset __getitem__() function
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        proposals = transposed_batch[2]
        index = transposed_batch[3]
        if self.cfg != None and self.cfg.MODEL.MULTI_TEACHER:
            trans_info = transposed_batch[4]
            return images, targets, proposals, index, trans_info
        return images, targets, proposals, index
