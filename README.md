# Pseudo R-CNN

### Requirements:
- Python3.5
- PyTorch 1.1
- torchvision 0.3.0
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0

# Installation

This implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as the original maskrcnn-benchmark.

Please check *INSTALL.md* for installation instructions. You may also want to see the original [README.md of maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/README.md).


# Training (Faster-ILOD)
```
./train_incremental.sh ./configs/10_10_new/e2e_faster_rcnn_R_50_C4_1x.yaml # (contains first step + incremental step)
```

**train_first_step.py**: normally train the first task (standard training). 

**train_incremental.py**: incrementally train the following tasks (knowledge distillation based training).

# Training (Pseudo R-CNN)
```
./train_pseudo_proto.sh ./configs/1010pseudo_proto/e2e_faster_rcnn_R_50_C4_1x.yaml
```


