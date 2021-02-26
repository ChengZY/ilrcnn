set -x

nr_gpu=1
ims_per_gpu=3 # 必须是1，否则会报错
ims_per_batch=`expr $nr_gpu \* $ims_per_gpu`
fpn_post_nms_top_n_train=`expr 1000 \* $ims_per_gpu`

# python -m torch.distributed.launch --nproc_per_node=$nr_gpu tools/train_first_step.py \
#     --config-file ./configs/e2e_faster_rcnn_R_50_C4_1x.yaml \
#     MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN $fpn_post_nms_top_n_train \
#     SOLVER.IMS_PER_BATCH $ims_per_batch

python tools/train_first_step.py \
    --config-file ./configs/e2e_faster_rcnn_R_50_C4_1x_debug.yaml \
    MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN $fpn_post_nms_top_n_train \
    SOLVER.IMS_PER_BATCH $ims_per_batch

# python tools/trim_detectron_model.py \
#    --pretrained_path /home/zhengkai/Faster-ILOD/incremental_learning_ResNet50_C4/RPN_19_classes_40k_steps_no_person/model_final.pth \
#    --save_path /home/zhengkai/Faster-ILOD/incremental_learning_ResNet50_C4/RPN_19_classes_40k_steps_no_person/model_trim_optimizer_iteration.pth