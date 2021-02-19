set -x
SESSION=1
EPOCH=10
CHECKPOINT=625

python test_net.py \
    --dataset pascal_voc \
    --net res101 \
    --checksession $SESSION \
    --checkepoch $EPOCH \
    --checkpoint $CHECKPOINT \
    --cuda
