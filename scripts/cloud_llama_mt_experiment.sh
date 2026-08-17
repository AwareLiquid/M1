#!/usr/bin/env bash
set -euo pipefail

# One-shot cloud experiment for Qwen/Llama + MT-LNN adapters.
#
# Intended for a fresh RunPod/Lambda-style Linux GPU box:
#   bash scripts/cloud_llama_mt_experiment.sh
#
# Override defaults with environment variables, for example:
#   MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct STEPS=2000 SEQ_LEN=2048 bash scripts/cloud_llama_mt_experiment.sh

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATASET="${DATASET:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
STEPS="${STEPS:-1000}"
MT_EVERY="${MT_EVERY:-4}"
MAX_PPL_BATCHES="${MAX_PPL_BATCHES:-50}"
NEEDLE_CONTEXTS="${NEEDLE_CONTEXTS:-1024 2048 4096}"
NEEDLE_DEPTHS="${NEEDLE_DEPTHS:-0.1 0.5 0.9}"
NEEDLE_SAMPLES="${NEEDLE_SAMPLES:-5}"
OUT_DIR="${OUT_DIR:-checkpoints/llama_mt_adapter}"
RESULT_DIR="${RESULT_DIR:-benchmarks/cloud_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

echo "== Environment =="
python - <<'PY'
import platform, torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("gpu_mem_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

echo
echo "== Installing Python dependencies =="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt accelerate safetensors

echo
echo "== Smoke tests =="
python -m pytest tests/test_llama_adapter.py -q

echo
echo "== Training MT adapter + LoRA =="
python train_llama_mt_adapter.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --dataset_config "${DATASET_CONFIG}" \
  --seq_len "${SEQ_LEN}" \
  --batch "${BATCH}" \
  --grad_accum "${GRAD_ACCUM}" \
  --steps "${STEPS}" \
  --mt_every "${MT_EVERY}" \
  --lora \
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "${RESULT_DIR}/train.log"

ADAPTER="$(ls -t "${OUT_DIR}"/llama_mt_adapter_*.pt | head -n 1)"
echo "Latest adapter: ${ADAPTER}" | tee "${RESULT_DIR}/adapter.txt"

echo
echo "== PPL ablation =="
python bench_llama_mt_ablation.py \
  --model "${MODEL}" \
  --adapters "${ADAPTER}" \
  --seq_len "${SEQ_LEN}" \
  --batch "${BATCH}" \
  --max_batches "${MAX_PPL_BATCHES}" \
  --out_json "${RESULT_DIR}/ppl_ablation.json" \
  2>&1 | tee "${RESULT_DIR}/ppl_ablation.log"

echo
echo "== Needle long-context ablation =="
python bench_llama_mt_needle.py \
  --model "${MODEL}" \
  --adapters "${ADAPTER}" \
  --context_lengths ${NEEDLE_CONTEXTS} \
  --depths ${NEEDLE_DEPTHS} \
  --samples "${NEEDLE_SAMPLES}" \
  --out_json "${RESULT_DIR}/needle.json" \
  2>&1 | tee "${RESULT_DIR}/needle.log"

echo
echo "== Done =="
echo "Results: ${RESULT_DIR}"
echo "Adapter: ${ADAPTER}"
