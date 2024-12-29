#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

PATH1=/nas/shared/pjlab_lingjun_landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/matrixcity/small_city/merge/pose/block_all/
PATH2=/nas/shared/pjlab_lingjun_landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/mipnerf360/bicycle/

EXPE_NAME=/cpfs01/user/liaozimu/code1/Grendel-GS/output/bike_n1d4
export PYTHONPATH=$PWD:$PYTHONPATH

conda activate gaussian_splatting

# Monitoring Settings
monitor_opts="--enable_timer \
    --end2end_time \
    --check_gpu_memory \
    --check_cpu_memory"


torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
        --bsz 4 \
        -s $PATH2 \
        --model_path ${EXPE_NAME} \
        --iterations 30000 \
        --log_interval 250 \
        $monitor_opts \
        --test_iterations 7000 15000 30000 \
        --save_iterations 7000 30000 \
        --eval


echo "rendering"

torchrun --standalone --nnodes=1 --nproc-per-node=4 render.py \
    -s ${PATH2} \
    --model_path ${EXPE_NAME} \
    --llffhold 8 \
    --skip_train \
    --bsz 4

echo "metric"
python metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}
