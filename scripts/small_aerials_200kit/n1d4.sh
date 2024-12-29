#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

PATH1=/nas/shared/pjlab_lingjun_landmarks/pjlab_lingjun_landmarks_hdd/yumulin_group/nerf_data/matrixcity/small_city/aerial/pose/block_all

NUM_GPUS=4
#EXPE_NAME=/cpfs01/user/liaozimu/code1/Grendel-GS/output/small_aerial_500kit_200kdensify_n1d$NUM_GPUS
export PYTHONPATH=$PWD:$PYTHONPATH

conda activate gaussian_splatting

# Monitoring Settings
monitor_opts="--enable_timer \
    --end2end_time \
    --check_gpu_memory \
    --check_cpu_memory"

denses=("0.002" "0.0016" "0.002" "0.0016")
grad_ths=("0.0001" "0.00008" "0.000075" "0.00006")
untils=("30000" "50000" "80000")
labels=("dense20gradth100" "dense16gradth80" "dense20gradth75" "dense16gradth60") 
iterations="7000 15000 30000 50000 65000 80000 100000 120000 150000 200000"
for i in {0..3};do
    for j in {0..2};do
        dense="${denses[$i]}"
        grad_th="${grad_ths[$i]}"
        until="${untils[$j]}"
        label="${labels[$i]}"
        EXPE_NAME=/oss/liaozimu/Grendel-GS/outputs/small_aerials/200kit_${label}_until${until}_n1d$NUM_GPUS
        echo "EXPERIMENT: densify_param=(${dense}, ${grad_ths}); densify_until=${until}"
        echo "training..."
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
                --test_iterations $iterations \
                --save_iterations 200000 \
                --densify_until_iter $until \
                --percent_dense $dense \
                --densify_grad_threshold $grad_th \
                --eval

        echo "rendering..."

        torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS render.py \
            -s ${PATH1} \
            --model_path ${EXPE_NAME} \
            --llffhold 8 \
            --skip_train \
            --bsz $NUM_GPUS 

        echo "metric..."
        python metrics_raw.py \
            --mode test \
        --model_paths ${EXPE_NAME}
    done
done
