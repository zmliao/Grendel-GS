#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
PATH1=/nas/shared/pjlab_lingjun_landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/matrixcity/small_city/merge/pose/block_all
PATH2=/nas/shared/pjlab-lingjun-landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/mipnerf360/bicycle/
PATH3=/nas/shared/pjlab_lingjun_landmarks/yumulin_group/nerf_data/matrixcity/small_city/aerial/pose/block_all

OUT_PATH=./output/small_all

conda activate gaussian_splatting
export PYTHONPATH=$PWD:$PYTHONPATH
nsys profile --stats=true --output my_report_bike --force-overwrite --trace=nvtx,cuda torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --bsz 8 -s $PATH1 -m $OUT_PATH --eval