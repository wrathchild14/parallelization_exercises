#!/bin/sh

# Locally:
# nvcc img_hist.cu -o image_hist
# ./image_hist lena.png

# On nsc:
module load CUDA/10.1.243-GCC-8.3.0
nvcc img_hist.cu -o image_hist
srun --reservation=fri -G1 -n1 image_hist lena.png
