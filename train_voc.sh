set -x
BATCH_SIZE=16
WORKER_NUMBER=16

python trainval_net.py \
	--dataset pascal_voc \
	--net res101 \
    --bs $BATCH_SIZE \
	--nw $WORKER_NUMBER \
	--lr 0.01 \
	--lr_decay_step 8 \
	--epochs 10 \
    --cuda \
    --mGPUs