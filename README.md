# Installation

This Faster ILOD implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as the original maskrcnn-benchmark.

Please check [INSTALL.md](https://github.com/CanPeng123/Faster-ILOD/blob/main/INSTALL.md) for installation instructions. You may also want to see the original [README.md of maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/README.md).


# Code Usage
**diff_result.py**: compare two configs' inference result into one AP output

# pretrain model for incremental learning
`normal setting`
**10+10**: "/data1/pbdata/data_zk/Faster-ILOD/incremental_learning_ResNet50_C4/10_10_new/source/model_final.pth"

**10+10**: "/data1/pbdata/data_zk/Faster-ILOD/incremental_learning_ResNet50_C4/10_10_new/source/model_trim_optimizer_iteration.pth"

**15+5**:

**15+5**:

**19+1**:

**19+1**:


`trick setting`
