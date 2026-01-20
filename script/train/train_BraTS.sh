#!/bin/bash
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29503 --num_processes=4 -m train.train_BraTS \
    --dataset "data/BraTS" \
    --checkpoints "final" \
    --gpu "0,1,2,3" \
    --batch-size 6 \
    --identifier "final" \
    --log_name "final" \
    --network "final" \
    --debug_mode
    # --resume