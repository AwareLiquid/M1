#!/bin/zsh
# Profile chain: wait for all three training sweeps (PIDs passed as args) to
# finish, then run the CPU/ONNX deployment profile on an otherwise-idle
# machine — latency numbers are meaningless under training load.
cd /Users/aricredemption/Projects/M1
for pid in "$@"; do
  while kill -0 $pid 2>/dev/null; do sleep 60; done
done
echo "[chain] all sweeps done at $(date)" >> benchmarks/results/chain.log
.venv/bin/python -u benchmarks/streaming_edge_profile.py \
  > benchmarks/results/streaming_edge_profile.log 2>&1
echo "[chain] profile done at $(date)" >> benchmarks/results/chain.log
