# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch

from maskrcnn_benchmark.utils.imports import import_file


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    print("=> Model required keys # {}, Loaded keys # {}".format(len(current_keys), len(loaded_keys)))

    distinct_prefix = []
    distinct_prefix2 = []
    # from ipdb import set_trace; set_trace()
    for x in current_keys:
        if x.startswith('roi_heads.box.feature_extractor'):
            print(x)
        pre = x.split('.')[0]
        if pre == 'roi_heads':
            if not (x.split('.')[2] in distinct_prefix2):
                distinct_prefix2.append(x.split('.')[2])
        if pre in distinct_prefix:
            continue
        distinct_prefix.append(pre)
    print("="*20)
    print(distinct_prefix)
    print(distinct_prefix2)
    # from ipdb import set_trace; set_trace()
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the loaded_key string, if it matches
    match_matrix = [len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue

        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]

        if model_state_dict[key].size() == loaded_state_dict[key_old].size(): # 若size相等则赋值
            model_state_dict[key] = loaded_state_dict[key_old]
        else:
            new_size = list(model_state_dict[key].size())
            print('model_serialization.py | new | size of {0}: {1}'.format(key, new_size))
            loaded_size = list(loaded_state_dict[key_old].size())
            print('model_serialization.py | loaded | size of {0}: {1}'.format(key, loaded_size))

            # print('model_serialization.py | new | value of {0}: {1}'.format(key, model_state_dict[key]))
            # print('model_serialization.py | loaded | value of {0}: {1}'.format(key_old, loaded_state_dict[key_old]))
            model_state_dict[key][:loaded_size[0]] = loaded_state_dict[key_old] # 复制old model的区间的参数过去
            # print('model_serialization.py | after loading, new | value of {0}: {1}'.format(key, model_state_dict[key]))

        # used for logging
        logger.info(log_str_template.format(key, max_size, key_old, max_size_loaded, tuple(loaded_state_dict[key_old].shape)))


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

# written by zhengkai for easy understand
def align_and_load(model_state_dict, loaded_state_dict):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    print("=> Model required keys # {}, Loaded keys # {}".format(len(current_keys), len(loaded_keys)))
    """
    loaded_keys: 
    - backbone.body.layerx.xx.xx  (A)
    - rpn.xx.xx (B)

    - roi_heads.box.feature_extractor (C)
    - roi_heads.box.predictor (D)

    current_keys:
    - backbone.body.layerx.xx.xx A
    - rpn.xx.xx                  B

    - roi_heads.box.feature_extractor C
    - roi_heads.box.predictor         D

    - roi_heads.box. backbone.xx       A
    - roi_heads.box. backbone_mean.xx  A
    - roi_heads.box. feature_extractor_mean C
    - roi_heads.box.proto_predictor   X
    - roi_heads.box.proto_predictor_mean X
    - roi_heads.box.queue               X
    - roi_heads.box.queue_iou           X
    - roi_heads.box.queue_label         X
    - roi_heads.box.queue_ptr           X
    """
    for new_key in current_keys:
        if new_key.startswith('backbone.') or new_key.startswith('rpn.'):
            if model_state_dict[new_key].size() == loaded_state_dict[new_key].size(): # 若size相等则赋值
                model_state_dict[new_key] = loaded_state_dict[new_key]
            else:
                new_size = list(model_state_dict[new_key].size())
                print('model_serialization.py | new | size of {0}: {1}'.format(new_key, new_size))
                loaded_size = list(loaded_state_dict[new_key].size())
                print('model_serialization.py | loaded | size of {0}: {1}'.format(new_key, loaded_size))

                model_state_dict[new_key][:loaded_size[0]] = loaded_state_dict[new_key] # 复制old model的区间的参数过去
        elif new_key.startswith('roi_heads.box.feature_extractor.') or new_key.startswith('roi_heads.box.predictor.'):
            """
            roi_heads.box.feature_extractor.head.layer4.0.bn1.bias
            roi_heads.box.feature_extractor.head.layer4.0.bn1.running_mean
            roi_heads.box.feature_extractor.head.layer4.0.bn1.running_var
            roi_heads.box.feature_extractor.head.layer4.0.bn1.weight
            roi_heads.box.feature_extractor.head.layer4.0.bn2.bias
            roi_heads.box.feature_extractor.head.layer4.0.bn2.running_mean
            ...
            roi_heads.box.feature_extractor.head.layer4.2.conv3.weight

            roi_heads.box.predictor.bbox_pred.bias
            roi_heads.box.predictor.bbox_pred.weight
            roi_heads.box.predictor.cls_score.bias
            roi_heads.box.predictor.cls_score.weight
            """
            if model_state_dict[new_key].size() == loaded_state_dict[new_key].size(): # 若size相等则赋值
                model_state_dict[new_key] = loaded_state_dict[new_key]
            else:
                new_size = list(model_state_dict[new_key].size())
                print('model_serialization.py | new | size of {0}: {1}'.format(new_key, new_size))
                loaded_size = list(loaded_state_dict[new_key].size())
                print('model_serialization.py | loaded | size of {0}: {1}'.format(new_key, loaded_size))

                model_state_dict[new_key][:loaded_size[0]] = loaded_state_dict[new_key] # 复制old model的区间的参数过去
        elif new_key.startswith('roi_heads.box.backbone.'):
            old_key = new_key[len('roi_heads.box.'):]
            if model_state_dict[new_key].size() == loaded_state_dict[old_key].size(): # 若size相等则赋值
                model_state_dict[new_key] = loaded_state_dict[old_key]
            else:
                new_size = list(model_state_dict[new_key].size())
                print('model_serialization.py | new | size of {0}: {1}'.format(new_key, new_size))
                loaded_size = list(loaded_state_dict[old_key].size())
                print('model_serialization.py | loaded | size of {0}: {1}'.format(new_key, loaded_size))

                model_state_dict[new_key][:loaded_size[0]] = loaded_state_dict[old_key] # 复制old model的区间的参数过去
        elif new_key.startswith('roi_heads.box.backbone_mean.'):
            old_key = 'backbone.' + new_key[len('roi_heads.box.backbone_mean.'):]
            if model_state_dict[new_key].size() == loaded_state_dict[old_key].size(): # 若size相等则赋值
                model_state_dict[new_key] = loaded_state_dict[old_key]
            else:
                new_size = list(model_state_dict[new_key].size())
                print('model_serialization.py | new | size of {0}: {1}'.format(new_key, new_size))
                loaded_size = list(loaded_state_dict[old_key].size())
                print('model_serialization.py | loaded | size of {0}: {1}'.format(new_key, loaded_size))

                model_state_dict[new_key][:loaded_size[0]] = loaded_state_dict[old_key] # 复制old model的区间的参数过去
        elif new_key.startswith('roi_heads.box.feature_extractor_mean.'):
            old_key = 'roi_heads.box.feature_extractor.' + new_key[len('roi_heads.box.feature_extractor_mean.'):]
            if model_state_dict[new_key].size() == loaded_state_dict[old_key].size(): # 若size相等则赋值
                model_state_dict[new_key] = loaded_state_dict[old_key]
            else:
                new_size = list(model_state_dict[new_key].size())
                print('model_serialization.py | new | size of {0}: {1}'.format(new_key, new_size))
                loaded_size = list(loaded_state_dict[old_key].size())
                print('model_serialization.py | loaded | size of {0}: {1}'.format(new_key, loaded_size))

                model_state_dict[new_key][:loaded_size[0]] = loaded_state_dict[old_key] # 复制old model的区间的参数过去
        elif new_key.startswith('roi_heads.box.proto_predictor') or \
            new_key.startswith('roi_heads.box.proto_predictor_mean') or \
            new_key.startswith('roi_heads.box.queue') or \
            new_key.startswith('roi_heads.box.queue_iou') or \
            new_key.startswith('roi_heads.box.queue_label') or \
            new_key.startswith('roi_heads.box.queue_ptr'):
            pass
        else:
            raise Exception("illegal keys {}".format(new_key))
            exit(0)


def load_state_dict(model, loaded_state_dict, proto_enable=False):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching

    for key, value in model_state_dict.items():
        # print('key : {0}'.format(key))
        if 'predictor.cls_score' in key or 'predictor.bbox_pred' in key:
            print('model_serialization.py | new created model | shape of {0}: {1}'.format(key, value.size()))
    for key, value in loaded_state_dict.items():
        # print('key : {0}'.format(key))
        if 'predictor.cls_score' in key or 'predictor.bbox_pred' in key:
            print('model_serialization.py | loaded model | shape of {0}: {1}'.format(key, value.size()))

    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    if not proto_enable:
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    else:
        align_and_load(model_state_dict, loaded_state_dict)
    # use strict loading
    model.load_state_dict(model_state_dict)
