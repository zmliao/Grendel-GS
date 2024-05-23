#!/bin/bash

# bench cross node communication scripr for node 0

# get NODE_COUNT and NODE_RANK from command line arguments
# NODE_COUNT=$1
# NODE_RANK=$2
# IP_ADDR=$3
# PORT=$4
NODE_COUNT=2
NODE_RANK=1
IP_ADDR="nid001164"
# $hostname
PORT=29500

torchrun \
    --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc-per-node=4 \
    --master_addr=$IP_ADDR --master_port=$PORT \
    bench_communication.py \
    --mode allreduce \
    --tensor-size 1024 \
    --num-iterations 10