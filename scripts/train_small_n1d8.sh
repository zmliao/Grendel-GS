#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

PATH1=/nas/shared/pjlab_lingjun_landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/matrixcity/small_city/merge/pose/block_all/
PATH2=/nas/shared/pjlab_lingjun_landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/mipnerf360/bicycle/

NUM_GPUS=8
EXPE_NAME=/cpfs01/user/liaozimu/code1/Grendel-GS/output/small_n1d$NUM_GPUS
export PYTHONPATH=$PWD:$PYTHONPATH

conda activate gaussian_splatting

# Monitoring Settings
monitor_opts="--enable_timer \
    --end2end_time \
    --check_gpu_memory \
    --check_cpu_memory"


torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py \
        --bsz $NUM_GPUS \
        -s $PATH1 \
        --model_path ${EXPE_NAME} \
        --iterations 200000 \
        --log_interval 2500 \
        $monitor_opts \
        --test_iterations 7000 15000 50000 200000 \
        --save_iterations 7000 15000 50000 200000 \
        --densify_until_iter 50000 \
        --percent_dense 0.0016 \
        --densify_grad_threshold 0.00008 \
        --eval


echo "rendering"

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS render.py \
    -s ${PATH1} \
    --model_path ${EXPE_NAME} \
    --llffhold 8 \
    --skip_train \
    --bsz $NUM_GPUS 

echo "metric"
python metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}
