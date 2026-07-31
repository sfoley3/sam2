#!/usr/bin/env bash
# Run sam2_propagate.py for all speakers, splitting across specified GPUs.
# Each worker queue processes its assigned speakers sequentially; workers run in parallel.
# Always launches 2 jobs (worker queues) per GPU.
# Usage: ./run_multispk_propagate.sh SESSION [GPU_IDS]
#   SESSION: data-collection session to propagate (e.g. D1A) — required
#   GPU_IDS: comma-separated GPU numbers, 1-2 GPUs (default: 0,1)
#   Examples: ./run_multispk_propagate.sh D1A 0,1
#             ./run_multispk_propagate.sh D1A 2

set -euo pipefail
cd "$(dirname "$0")"

DATASET="longitudinal"

# Parse data-collection session from first argument (required)
SESSION="${1:-}"
if [[ -z "$SESSION" ]]; then
    echo "ERROR: SESSION is required (e.g. D1A)." >&2
    echo "Usage: $0 SESSION [GPU_IDS]" >&2
    exit 1
fi

# Parse GPU IDs from second argument, default to "0,1"
GPU_ARG="${2:-0,1}"
IFS=',' read -ra GPUS <<< "$GPU_ARG"
NUM_GPUS=${#GPUS[@]}

# Always run 2 jobs (worker queues) per GPU
WORKERS_PER_GPU=2
NUM_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

ALL_SPEAKERS=(ID01 ID02 ID03 ID07 ID08 ID09 ID10
                 ID11 ID12 ID13 ID14 ID16 ID17 ID18 ID20 ID21)

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

echo "=== Propagation: ${#ALL_SPEAKERS[@]} speakers, session ${SESSION}, on ${NUM_GPUS} GPUs (${GPU_ARG}) with ${NUM_WORKERS} worker queues ==="
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
            echo "[GPU ${gpu_id}] Starting $spk ($SESSION) ..."
            python sam2_propagate.py \
                --spk "$spk" --dataset "$DATASET" --data-session "$SESSION" \
                --gpus "$gpu_id" --jobs 1 2>&1 | sed "s/^/[GPU${gpu_id} $spk] /"
            echo "[GPU ${gpu_id}] Finished $spk"
        done
    ) &
    PIDS+=($!)
done

echo "Waiting for all ${NUM_WORKERS} worker queues to finish..."
wait "${PIDS[@]}"

echo ""
echo "=== All speakers done ==="
