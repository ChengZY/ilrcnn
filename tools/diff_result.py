"""
usage:
    python tools/diff_result.py --target 10_10_new --base 15_5_pseudo4_new_lownms
"""
import argparse
import os
import copy

YML_ROOT = '/data1/pbdata/data_zk/Faster-ILOD/incremental_learning_ResNet50_C4'
RESULT_PATH = YML_ROOT + '/{}/target/inference/voc_2007_test/result.txt'

CATEGORIES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def parse_txt(txt):
    ap_dict = {"mAP": 0.}
    for x in CATEGORIES:
        ap_dict[x] = 0.
    with open(txt) as f:
        buff = f.readlines()
        buff = [x.strip("\n") for x in buff]
        # for each class AP
        for i in range(len(buff)):
            x = buff[i]
            str_list = x.split(':')
            catname = str_list[0].strip(' ')
            ap = float(str_list[1].strip(' ')) * 100
            ap_dict[catname] = round(ap, 2)
            # from ipdb import set_trace; set_trace()
    
    return ap_dict
        

def main():
    parser = argparse.ArgumentParser(description="Evaluation Result Diff")
    parser.add_argument(
        "--target",
        default="",
        metavar="CFG1",
        help="target file",
        type=str,
    )
    parser.add_argument(
        "--base",
        default="",
        metavar="CFG2",
        help="base file to be compared with",
        type=str,
    )
    args = parser.parse_args()

    txt1 = RESULT_PATH.format(args.target)
    txt2 = RESULT_PATH.format(args.base)
    print(txt1)
    print(txt2)
    assert os.path.isfile(txt1), "Ensure {} exists".format(txt1)
    assert os.path.isfile(txt2), "Ensure {} exists".format(txt2)

    tgt_dict = parse_txt(txt1)
    base_dict = parse_txt(txt2)

    assert tgt_dict.keys() == base_dict.keys()

    result_str = ""
    for x in tgt_dict.keys():
        if x in ['mAP', 'old mAP', 'new mAP']:
            if tgt_dict[x]-base_dict[x] > 0:
                result_str += "{}: {:.2f} (+{:.2f})\n".format(x, tgt_dict[x], tgt_dict[x]-base_dict[x])
            else:
                result_str += "{}: {:.2f} ({:.2f})\n".format(x, tgt_dict[x], tgt_dict[x]-base_dict[x])
        else:
            if tgt_dict[x]-base_dict[x] > 0:
                result_str += "{:<16}: {:.2f} (+{:.2f})\n".format(x, tgt_dict[x], tgt_dict[x]-base_dict[x])
            else:
                result_str += "{:<16}: {:.2f} ({:.2f})\n".format(x, tgt_dict[x], tgt_dict[x]-base_dict[x])
    print(result_str)


if __name__ == "__main__":
    main()