#!/bin/bash

# Automatically set TRITON_PTXAS_PATH using the output of the Python command
TRITON_PTXAS_PATH=$(python -c "import triton;print(triton.common.backend.path_to_ptxas())")
export TRITON_PTXAS_PATH

# Function to reset GPU and clear VRAM cache
reset_gpu() {
    echo "Resetting GPU to clear VRAM cache..."
    sudo nvidia-smi --gpu-reset
    sleep 5  # Give some time for the GPU to reset
}

# Function to get VRAM usage
get_vram_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}'
}

# Function to extract the latest elapsed time from tqdm output
extract_elapsed_time() {
    local log_file=$1
    grep -oP '100%.*\[\K[0-9]+:[0-9]+' "$log_file" | tail -1
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

    mfu=$(echo "scale=6; $total_flops / ($elapsed_time * $theoretical_peak_flops)" | bc -l 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo "Error calculating MFU with bc."
        mfu="N/A"
    fi
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
    local config_name=$8

    echo "$batch_size,$parameters,$elapsed_time,$total_flops,$mfu,$compile_flag,$config_name" >> "$csv_file"
}

# Define a function to run the commands and capture logs
run_and_log() {
    local config_file=$1
    local name=$2
    local batch_size=$3
    local compile_flag=$4
    local csv_file=$5
    local config_name=$6

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
    elif [[ "$elapsed_time" =~ ([0-9]+):([0-9]+):([0-9]+) ]]; then
        hours=${BASH_REMATCH[1]}
        minutes=${BASH_REMATCH[2]}
        seconds=${BASH_REMATCH[3]}
        elapsed_time=$(echo "$hours * 3600 + $minutes * 60 + $seconds" | bc)
    else
        echo "Error extracting elapsed time from log file."
        elapsed_time=0
    fi

    # Extract parameters from log file
    parameters=$(extract_parameters_from_log "$log_file")

    # Calculate total FLOPs
    total_flops=$(grep -oP 'Total FLOPs: \K[0-9.]+(?= TFLOPs)' "$log_file" | awk '{print $1 * 1e12}')
    echo "Total FLOPs: $total_flops"

    # Calculate MFU
    if [ "$elapsed_time" -gt 0 ]; then
        mfu=$(calculate_mfu "$total_flops" "$elapsed_time")
        echo "Estimated MFU: $mfu"
    else
        echo "Error: Elapsed time is zero or not valid. Cannot calculate MFU."
        mfu="N/A"
    fi

    # Log results to CSV
    log_to_csv "$csv_file" "$batch_size" "$parameters" "$elapsed_time" "$total_flops" "$mfu" "$compile_flag" "$config_name"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# CSV file to store results
csv_file="results.csv"

# Check if CSV file already exists
if [ -f "$csv_file" ]; then
    # Back up the existing CSV file
    mv "$csv_file" "${csv_file}.bak"
    echo "Existing CSV file backed up as ${csv_file}.bak"
fi

# Create a new CSV file with headers
echo "Batch Size,Parameters,Elapsed Time (s),Total FLOPs,MFU,Compile Flag,Config" > "$csv_file"

# Run commands for config_oxford_flowers_big.json
reset_gpu
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 32 "" "$csv_file" "natten"

reset_gpu
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 32 "--compile" "$csv_file" "natten"

reset_gpu
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 50 "" "$csv_file" "natten"

reset_gpu
run_and_log "configs/config_oxford_flowers_big.json" "flowers_demo_001" 50 "--compile" "$csv_file" "natten"

# Run commands for config_oxford_flowers_shifted_window_big.json
reset_gpu
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 32 "" "$csv_file" "shifted_window"

reset_gpu
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 32 "--compile" "$csv_file" "shifted_window"

reset_gpu
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 50 "" "$csv_file" "shifted_window"

reset_gpu
run_and_log "configs/config_oxford_flowers_shifted_window_big.json" "flowers_demo_001" 50 "--compile" "$csv_file" "shifted_window"

# Output summary
echo "Training completed. Results have been logged to $csv_file."

# Generate plots from the main results.csv file
python generate_plots.py
