#!/usr/bin/env bash
# Run sam2_propagate.py for all speakers, splitting across specified GPUs.
# Each worker queue processes its assigned speakers sequentially; workers run in parallel.
# When only one GPU is provided, launch two workers on that GPU.
# Usage: ./run_multispk_propagate.sh [GPU_IDS]
#   GPU_IDS: comma-separated GPU numbers (default: 0,1)
#   Examples: ./run_multispk_propagate.sh 0,1
#             ./run_multispk_propagate.sh 2,3,4

set -euo pipefail
cd "$(dirname "$0")"

DATASET="prompt"

# Parse GPU IDs from first argument, default to "0,1"
GPU_ARG="${1:-0,1}"
IFS=',' read -ra GPUS <<< "$GPU_ARG"
NUM_GPUS=${#GPUS[@]}

WORKERS_PER_GPU=1
if (( NUM_GPUS == 1 )); then
    WORKERS_PER_GPU=3
fi
NUM_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

ALL_SPEAKERS=(spk2 spk3 spk4 spk5 spk6 spk7 spk8 spk9 spk10)

# Distribute speakers round-robin across worker queues
declare -a GPU_SPEAKERS
declare -a WORKER_GPU_IDS
for ((g=0; g<NUM_WORKERS; g++)); do
    WORKER_GPU_IDS[$g]="${GPUS[$((g % NUM_GPUS))]}"
    GPU_SPEAKERS[$g]=""
done
for ((i=0; i<${#ALL_SPEAKERS[@]}; i++)); do
    g=$((i % NUM_WORKERS))
    GPU_SPEAKERS[$g]="${GPU_SPEAKERS[$g]} ${ALL_SPEAKERS[$i]}"
done

echo "=== Propagation: ${#ALL_SPEAKERS[@]} speakers on ${NUM_GPUS} GPUs (${GPU_ARG}) with ${NUM_WORKERS} worker queues ==="
for ((g=0; g<NUM_WORKERS; g++)); do
    echo "Worker $((g + 1)) on GPU ${WORKER_GPU_IDS[$g]}:${GPU_SPEAKERS[$g]}"
done
echo ""

# Launch worker queues in parallel
PIDS=()
for ((g=0; g<NUM_WORKERS; g++)); do
    gpu_id=${WORKER_GPU_IDS[$g]}
    read -ra spk_list <<< "${GPU_SPEAKERS[$g]}"
    (
        for spk in "${spk_list[@]}"; do
            echo "[GPU ${gpu_id}] Starting $spk ..."
            CUDA_VISIBLE_DEVICES=$gpu_id python sam2_propagate.py \
                --spk "$spk" --dataset "$DATASET" --jobs 1 2>&1 | sed "s/^/[GPU${gpu_id} $spk] /"
            echo "[GPU ${gpu_id}] Finished $spk"
        done
    ) &
    PIDS+=($!)
done

echo "Waiting for all ${NUM_WORKERS} worker queues to finish..."
wait "${PIDS[@]}"

echo ""
echo "=== All speakers done ==="
