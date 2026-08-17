"""MT-LNN Ablation Study - Kaggle Notebook

Systematically tests different adapter configurations to understand performance drivers.

Tests 4 ablation groups:
1. Layer interval: every 2 vs 4 vs 8
2. LoRA rank: 4 vs 8 vs 16
3. Adapter type: MT-only vs LoRA-only vs MT+LoRA
4. Protofilaments: 8 vs 13 vs 21

Wall: ~8h on T4 for all groups (200 steps each × ~12 configs).
"""

# Cell 1: Setup
import subprocess, sys

# Pin torch for sm_60 + sm_75 compatibility
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
    'torch==2.4.1', 'torchvision==0.19.1',
    '--index-url', 'https://download.pytorch.org/whl/cu121'
])
print('torch pinned to 2.4.1+cu121')

import sys as _s
if 'torch' in _s.modules:
    import os, signal
    os.kill(os.getpid(), signal.SIGTERM)

# Cell 2: Clone M1 repo
import os, subprocess, torch

REPO = 'https://github.com/AwareLiquid/M1.git'
DIR = '/kaggle/working/M1'

if not os.path.exists(DIR):
    subprocess.check_call(['git', 'clone', '--depth', '1', REPO, DIR])

os.chdir(DIR)
subprocess.check_call(['git', 'log', '-1', '--oneline'])

cap = torch.cuda.get_device_capability(0)
print(f'GPU: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}')

# Cell 3: Install dependencies
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    '-r', 'requirements.txt', 'accelerate', 'peft', 'datasets'
])

# Cell 4: Run ablations
import json
import os
from pathlib import Path

# Import after installation
from scripts.run_ablations import ABLATION_GROUPS, train_one_ablation, save_results, print_summary_table

# Arguments
class Args:
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    steps = 200
    batch = 1
    seq_len = 384
    grad_accum = 8
    lr = 2e-4
    weight_decay = 0.01
    grad_clip = 1.0
    log_every = 20
    device = "cuda"
    out_dir = "/kaggle/working/ablations"

args = Args()

# Select which groups to run (change as needed)
# Options: 'layer_interval', 'lora_rank', 'adapter_type', 'protofilaments', 'all'
GROUP_TO_RUN = 'adapter_type'  # Start with adapter_type (MT vs LoRA vs MT+LoRA)

if GROUP_TO_RUN == 'all':
    configs = [c for group in ABLATION_GROUPS.values() for c in group]
else:
    configs = ABLATION_GROUPS[GROUP_TO_RUN]

print(f'Running {len(configs)} ablations from group: {GROUP_TO_RUN}')
print(f'Model: {args.model}, Steps: {args.steps}, Device: {args.device}')
print()

# Run ablations
results = []
for cfg in configs:
    try:
        result = train_one_ablation(cfg, args)
        results.append(result)
    except Exception as e:
        print(f'\\nERROR in {cfg.name}: {e}')
        import traceback
        traceback.print_exc()
        continue

# Save results
if results:
    out_path = os.path.join(args.out_dir, f'ablation_{GROUP_TO_RUN}_results.json')
    save_results(results, out_path)
    print_summary_table(results)

# Cell 5: Archive results
import shutil

archive = shutil.make_archive(
    '/kaggle/working/ablation_results',
    'zip',
    args.out_dir
)
print(f'Archive: {archive}')
print('Download this to add to M1 repo as artifacts/ablations/')
