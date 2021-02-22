#!/bin/sh
echo "CUDA_VISIBLE_DEVICES=XXX ./runall.sh SESSION_NUM config_name"
SESSION=$1
CFG=$2

cmd="python3 trainval_net.py --use_tfboard --session ${SESSION} --conf ${CFG} --vis"
echo $cmd
$cmd # > logs/${SESSION}_train.log
echo $cmd

# cmd="python3 test_net.py --s ${SESSION} --no_repr --conf ${CFG}"
# echo $cmd
# $cmd > logs/${SESSION}_test-norepr.log
# tail -n11 logs/${SESSION}_test-norepr.log
# echo $cmd
