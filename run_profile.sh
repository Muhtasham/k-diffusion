#!/bin/bash

# Set the TRITON_PTXAS_PATH
export TRITON_PTXAS_PATH="/home/ubuntu/.local/lib/python3.10/site-packages/triton/common/../third_party/cuda/bin/ptxas"

# Define a function to run the commands and capture logs
run_and_log() {
    local config_file=$1
    local name=$2
    local batch_size=$3
    local compile_flag=$4

    if [ "$compile_flag" == "--compile" ]; then
        compile_text="_compile"
    else
        compile_text=""
    fi

    log_file="logs/${name}_${batch_size}${compile_text}.log"
    vram_log_file="logs/${name}_${batch_size}${compile_text}_vram.log"

    echo "Running: python train.py --config $config_file --name $name --evaluate-n 0 --batch-size $batch_size --sample-n 36 --mixed-precision bf16 $compile_flag"
    
    # Log initial VRAM usage
    echo "Initial VRAM usage:" | tee -a $vram_log_file
    nvidia-smi | tee -a $vram_log_file
    
    # Run the training command and log the output
    python train.py --config $config_file --name $name --evaluate-n 0 --batch-size $batch_size --sample-n 36 --mixed-precision bf16 $compile_flag | tee $log_file
    
    # Log final VRAM usage
    echo "Final VRAM usage:" | tee -a $vram_log_file
    nvidia-smi | tee -a $vram_log_file
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Run commands for config_oxford_flowers_big.json
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 32 ""
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 32 "--compile"
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 50
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 50 "--compile"

# Run commands for config_oxford_flowers_shifted_window_big.json
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 32 ""
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 32 "--compile"
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 50
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 50 "--compile"
