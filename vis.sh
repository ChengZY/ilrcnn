#!/usr/bin/zsh


SRC_YML="./configs/10_10/e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml"
TAT_YML="./configs/10_10/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml"
echo $SRC_YML
echo $TAT_YML

set -x

python tools/vis_2model.py \
    --src-file $SRC_YML \
    --tat-file $TAT_YML