set -x
BATCH_SIZE=16
WORKER_NUMBER=16

python trainval_net.py \
	--dataset coco \
	--net res101 \
    --bs $BATCH_SIZE \
	--nw $WORKER_NUMBER \
	--lr 0.01 \
	--lr_decay_step 4 \
	--epochs 6 \
    --cuda \
    --mGPUs