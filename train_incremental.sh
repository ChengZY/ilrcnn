#!/usr/bin/zsh
# CUDA_VISIBLE_DEVICES=3 ./train.sh ./configs/19_1_again/e2e_faster_rcnn_R_50_C4_1x.yaml
echo "train.sh YML"
nr_gpu=1
ims_per_gpu=1 # 必须是1，否则会报错
ims_per_batch=`expr $nr_gpu \* $ims_per_gpu`
fpn_post_nms_top_n_train=`expr 1000 \* $ims_per_gpu`

YML=$1 # ./configs/10_10/e2e_faster_rcnn_R_50_C4_1x.yaml
echo "original file: $YML"
length=${#YML}
endidx=`expr $length - 5`

# echo "enndidx: $endidx"
echo ${YML:0:37}
SRC_YML=${YML:0:${endidx}}"_Source_model.yaml"
TAT_YML=${YML:0:${endidx}}"_Target_model.yaml"
echo "source file: $SRC_YML"
echo "target file: $TAT_YML"

set -x
# python -m torch.distributed.launch --nproc_per_node=$nr_gpu tools/train_first_step.py \
#     --config-file ./configs/e2e_faster_rcnn_R_50_C4_1x.yaml \
#     MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN $fpn_post_nms_top_n_train \
#     SOLVER.IMS_PER_BATCH $ims_per_batch

python tools/train_first_step.py \
    --config-file $YML \
    MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN $fpn_post_nms_top_n_train \
    SOLVER.IMS_PER_BATCH $ims_per_batch

python tools/trim_detectron_model.py \
   --config-file $YML

python tools/train_incremental.py --src-file $SRC_YML --tat-file $TAT_YML