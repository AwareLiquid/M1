"""M1 adapter SFT (bilingual instruction tuning).

Trains the MT-LNN residual adapter + LoRA on TinyLlama-1.1B-Chat using the new
`--task sft` path: (instruction, input, output) triples are formatted through
the chat template and the loss is masked to the assistant completion only. This
is the post-training stage the deployed step-1000 adapter skipped (it was plain
WikiText causal-LM, so it could continue English text but could not follow
instructions, reason, or answer in Chinese).

Datasets: tatsu-lab/alpaca (EN) + shibing624/alpaca-zh (ZH) -> bilingual mix.
Output : /kaggle/working/out/llama_mt_adapter_*.pt  (download as kernel output,
         drop into checkpoints/llama_mt_adapter/ and point ADAPTER_CKPT at it).

Run as a Kaggle script kernel (GPU T4, internet on). ~1-2h.
"""

import os
import subprocess
import sys

REPO = "https://github.com/everest-an/M1.git"
DIR = "/kaggle/working/M1"

if not os.path.exists(DIR):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, DIR])
os.chdir(DIR)
subprocess.check_call(["git", "log", "-1", "--oneline"])

# Kaggle's current default torch (2.10+cu128) dropped CUDA kernels for older
# GPUs such as the P100 (sm_60) that sessions are sometimes assigned, causing
# "CUDA error: no kernel image is available for execution on the device". Install
# a torch whose wheels ship sm_60..sm_90 so the job runs on whatever GPU Kaggle
# hands out (P100 / T4). Install torch LAST so it wins dependency resolution.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "accelerate", "peft", "datasets", "transformers",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
])

import torch  # noqa: E402

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
else:
    print("GPU: CPU")

os.makedirs("/kaggle/working/out", exist_ok=True)

cmd = [
    sys.executable, "train_llama_mt_adapter.py",
    "--task", "sft",
    "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "--dataset", "tatsu-lab/alpaca,shibing624/alpaca-zh",
    "--dataset_config", "",            # SFT sets need no HF config name
    "--steps", "3000",
    "--batch", "2",
    "--grad_accum", "8",
    "--seq_len", "768",
    "--lr", "2e-4",
    "--save_every", "1000",
    "--lora",
    "--out_dir", "/kaggle/working/out",
    # MT adapter recipe -- identical to the deployed checkpoint so the result is
    # a drop-in replacement (same graph, better training).
    "--mt_every", "4", "--mt_proto", "13", "--mt_scales", "5",
    "--lora_r", "8", "--lora_alpha", "16", "--lora_targets", "q_proj,k_proj,v_proj,o_proj",
]
print("RUN:", " ".join(cmd))
subprocess.check_call(cmd)

print("=== output checkpoints ===")
subprocess.check_call(["ls", "-la", "/kaggle/working/out"])
print("SFT_KERNEL_DONE")
