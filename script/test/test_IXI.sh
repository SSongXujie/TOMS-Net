#!/bin/bash
export NCCL_P2P_DISABLE=1
# --main_process_port 29502 
accelerate launch --main_process_port 29502  --num_processes=3 -m test.test_IXI \
    --dataset "data/IXI" \
    --output-dir ./test_results \
    --checkpoints "final/best_model/" \
    --batch-size 6 \
    --gpu "0,1,2" \
    --save_results_name "final" \
    --network "final" \
   