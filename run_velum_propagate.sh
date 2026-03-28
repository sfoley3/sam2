#!/usr/bin/env bash
# Run sam2_propagate.py for all velum_study speakers, splitting across GPUs 0 and 1.
# Each speaker uses both GPUs (--gpus 0,1) to distribute its videos, and
# speakers are processed sequentially to avoid OOM.
# To run two speakers in parallel (one per GPU), we launch pairs in the background.

set -euo pipefail
cd "$(dirname "$0")"

DATASET="velum"
SPEAKERS=(spk1 spk2 spk4 spk5 spk6 spk7 spk8)

# Split speakers into two groups for GPU 0 and GPU 1
GPU0_SPEAKERS=(spk1 spk2 spk4)
GPU1_SPEAKERS=(spk5 spk6 spk7 spk8)

echo "=== Velum propagation: ${#SPEAKERS[@]} speakers on GPUs 0,1 ==="
echo "GPU 0: ${GPU0_SPEAKERS[*]}"
echo "GPU 1: ${GPU1_SPEAKERS[*]}"
echo ""

# Run both GPU queues in parallel
(
    for spk in "${GPU0_SPEAKERS[@]}"; do
        echo "[GPU 0] Starting $spk ..."
        CUDA_VISIBLE_DEVICES=0 python sam2_propagate.py \
            --spk "$spk" --dataset "$DATASET" 2>&1 | sed "s/^/[GPU0 $spk] /"
        echo "[GPU 0] Finished $spk"
    done
) &
pid0=$!

(
    for spk in "${GPU1_SPEAKERS[@]}"; do
        echo "[GPU 1] Starting $spk ..."
        CUDA_VISIBLE_DEVICES=1 python sam2_propagate.py \
            --spk "$spk" --dataset "$DATASET" 2>&1 | sed "s/^/[GPU1 $spk] /"
        echo "[GPU 1] Finished $spk"
    done
) &
pid1=$!

echo "Waiting for both GPU queues to finish..."
wait $pid0 $pid1

echo ""
echo "=== All speakers done ==="
