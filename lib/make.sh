#!/usr/bin/env bash

# export CXXFLAGS="-std=c++11"
# export CFLAGS="-std=c99"

# CUDA_PATH=/usr/local/cuda/bin
# export PATH=$PATH:${CUDA_PATH}

export CUDA_PATH=/usr/local/cuda/
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

PYTHON=python3

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

${PYTHON} setup.py build_ext --inplace
rm -rf build

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
${PYTHON} build.py

# compile roi_pooling
cd ../../
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
${PYTHON} build.py

# compile roi_align
cd ../../
cd model/roi_align/src
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
${PYTHON} build.py

# compile roi_crop
cd ../../
cd model/roi_crop/src
echo "Compiling roi crop kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
${PYTHON} build.py
