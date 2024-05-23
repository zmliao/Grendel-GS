

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --iterations 80000 \
    --log_interval 250 \
    --log_folder experiments/rubble_mp4_2_30k_sbc \
    --model_path experiments/rubble_mp4_2_30k_sbc \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 1 \
    --bsz 1 \
    --densify_grad_threshold 0.0001 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 \
    --save_iterations 7000 15000 30000 50000 80000 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory