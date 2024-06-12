#!/bin/bash

# Set the TRITON_PTXAS_PATH
export TRITON_PTXAS_PATH="/home/ubuntu/.local/lib/python3.10/site-packages/triton/common/../third_party/cuda/bin/ptxas"

# Function to get VRAM usage
get_vram_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}

# Function to extract the latest elapsed time from tqdm output
extract_elapsed_time() {
    local log_file=$1
    grep '100%' "$log_file" | tail -1 | awk -F'\\[|\\]' '{print $4}'
}

# Function to extract parameters from log file
extract_parameters_from_log() {
    local log_file=$1
    grep 'Parameters:' "$log_file" | awk '{print $2}' | tr -d ','
}

# Function to calculate MFU
calculate_mfu() {
    local total_flops=$1
    local elapsed_time=$2
    local theoretical_peak_flops=312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS

    mfu=$(echo "$total_flops / ($elapsed_time * $theoretical_peak_flops)" | bc -l)
    echo "$mfu"
}

# Function to log results to CSV
log_to_csv() {
    local csv_file=$1
    local batch_size=$2
    local parameters=$3
    local elapsed_time=$4
    local total_flops=$5
    local mfu=$6
    local compile_flag=$7

    echo "$batch_size,$parameters,$elapsed_time,$total_flops,$mfu,$compile_flag" >> "$csv_file"
}

# Define a function to run the commands and capture logs
run_and_log() {
    local config_file=$1
    local name=$2
    local batch_size=$3
    local compile_flag=$4
    local csv_file=$5

    if [ "$compile_flag" == "--compile" ]; then
        compile_text="_compile"
    else
        compile_text=""
    fi

    log_file="logs/${name}_${batch_size}${compile_text}.log"
    vram_log_file="logs/${name}_${batch_size}${compile_text}_vram.log"

    echo "Running: python train.py --config $config_file --name $name --evaluate-n 0 --batch-size $batch_size --sample-n 36 --mixed-precision bf16 $compile_flag"
    
    # Run the training command and log the output, including tqdm output
    (python train.py --config $config_file --name $name --evaluate-n 0 --batch-size $batch_size --sample-n 36 --mixed-precision bf16 $compile_flag 2>&1 | tee $log_file) &

    # Get the PID of the training process
    train_pid=$!

    # Monitor VRAM usage periodically
    while kill -0 $train_pid 2> /dev/null; do
        vram_usage=$(get_vram_usage)
        echo "VRAM usage: $vram_usage MB" | tee -a $vram_log_file
        sleep 60  # Adjust the sleep duration as needed
    done

    # Wait for the training process to finish
    wait $train_pid

    # Extract elapsed time from the log file
    elapsed_time=$(extract_elapsed_time "$log_file")
    echo "Elapsed time: $elapsed_time"

    # Convert elapsed time to seconds
    if [[ "$elapsed_time" =~ ([0-9]+):([0-9]+) ]]; then
        minutes=${BASH_REMATCH[1]}
        seconds=${BASH_REMATCH[2]}
        elapsed_time=$(echo "$minutes * 60 + $seconds" | bc)
    fi

    # Extract parameters from log file
    parameters=$(extract_parameters_from_log "$log_file")

    # Calculate total FLOPs (example value, replace with actual extraction logic)
    total_flops=$(grep -oP 'Total FLOPs: \K[0-9.]+(?= TFLOPs)' "$log_file" | awk '{print $1 * 1e12}')
    echo "Total FLOPs: $total_flops"

    # Calculate MFU
    mfu=$(calculate_mfu "$total_flops" "$elapsed_time")
    echo "Estimated MFU: $mfu"

    # Log results to CSV
    log_to_csv "$csv_file" "$batch_size" "$parameters" "$elapsed_time" "$total_flops" "$mfu" "$compile_flag"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# CSV file to store results
csv_file="results.csv"
echo "Batch Size,Parameters,Elapsed Time (s),Total FLOPs,MFU,Compile Flag" > "$csv_file"

# Run commands for config_oxford_flowers_big.json
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 32 "" "$csv_file"
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 32 "--compile" "$csv_file"
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 50 "" "$csv_file"
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 50 "--compile" "$csv_file"

# Run commands for config_oxford_flowers_shifted_window_big.json
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 32 "" "$csv_file"
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 32 "--compile" "$csv_file"
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 50 "" "$csv_file"
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 50 "--compile" "$csv_file"

# Output summary
echo "Profilings completed. Results have been logged to $csv_file."