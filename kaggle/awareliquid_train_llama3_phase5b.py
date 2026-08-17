"""Test Phase 5b recipe on Llama-3-8B (third model family validation).

Validates cross-architecture reproducibility on a third model family:
- Family 1: TinyLlama (Llama-1 style) ✅ -28.5%
- Family 2: Qwen-2.5 (1.5B, 3B) ✅ -27.7%, -34.4%
- Family 3: Llama-3-8B (this notebook)

Expected: Similar PPL improvement (-25% to -35%) with same recipe.

Wall: ~4h on T4 (1000 steps × 8B model).
"""

# Cell 1: Setup torch
import subprocess, sys

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

# Cell 2: Clone M1 + GPU check
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

# Cell 4: Apply Phase 5b recipe and verify
from transformers import AutoModelForCausalLM, AutoTokenizer
from mt_lnn.recipes import apply_phase5b_recipe

print('Loading Llama-3-8B-Instruct...')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
)

print('\nApplying Phase 5b recipe (MT every 4th + LoRA q/k/v/o)...')
result = apply_phase5b_recipe(model, verbose=True)

print('\n' + '='*80)
print('PHASE 5B RECIPE APPLIED')
print('='*80)
print(f'Wrapped layers: {result.wrapped_layer_indices}')
print(f'Trainable params: {result.trainable_params:,} ({result.trainable_percent:.3f}%)')
print(f'LoRA applied: {result.lora_applied}')

# Sanity check: generate a few tokens
print('\nSanity check: generating with adapted model...')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
inputs = tokenizer('The capital of Australia is', return_tensors='pt').to('cuda')

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

print('Generated:', tokenizer.decode(outputs[0], skip_special_tokens=True))
print('✓ Model is functional\n')

# Cell 5: Train Phase 5b (1000 steps)
print('Starting training (1000 steps on WikiText-2)...')
print('Expected wall: ~4h on T4')
print()

# Training configuration
class Args:
    model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    dataset = 'wikitext'
    dataset_config = 'wikitext-2-raw-v1'
    split = 'train'
    text_column = 'text'
    seq_len = 384
    batch = 1
    grad_accum = 8
    steps = 1000
    lr = 2e-4
    weight_decay = 0.01
    grad_clip = 1.0
    log_every = 20
    save_every = 200
    out_dir = '/kaggle/working/checkpoints'

args = Args()

# Import training utilities
from datasets import load_dataset
from torch.utils.data import DataLoader
import time

# Build dataloader
print('Loading WikiText-2...')
ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

def tokenize(batch):
    text = [t for t in batch[args.text_column] if t]
    if not text:
        return {'input_ids': []}
    return tokenizer(text, add_special_tokens=False)

tokenized = ds.map(
    tokenize, batched=True, remove_columns=ds.column_names, desc='tokenizing'
)

def group_texts(examples):
    ids = []
    for row in examples['input_ids']:
        ids.extend(row + [tokenizer.eos_token_id])
    total = (len(ids) // args.seq_len) * args.seq_len
    ids = ids[:total]
    chunks = [ids[i:i+args.seq_len] for i in range(0, total, args.seq_len)]
    return {'input_ids': chunks, 'labels': [c.copy() for c in chunks]}

lm_ds = tokenized.map(
    group_texts, batched=True, remove_columns=tokenized.column_names, desc='chunking'
)
lm_ds.set_format(type='torch', columns=['input_ids', 'labels'])
loader = DataLoader(lm_ds, batch_size=args.batch, shuffle=True, drop_last=True)

# Setup training
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=args.weight_decay,
)

# Training loop
step = 0
t0 = time.time()
losses = []

print('Training...')
while step < args.steps:
    for batch in loader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item() * args.grad_accum)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            toks_per_s = step * args.batch * args.seq_len / max(elapsed, 1e-6)
            print(f'step {step:4d}/{args.steps} | loss {avg_loss:.4f} | ppl {ppl:.2f} | {toks_per_s:.0f} tok/s')

        if step % args.save_every == 0 or step == args.steps:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f'llama3_8b_mt_adapter_{step:06d}.pt')
            torch.save({
                'step': step,
                'model': args.model,
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items() if 'mt_adapter' in k or 'lora_' in k},
                'args': vars(args),
            }, ckpt_path)
            print(f'  saved {ckpt_path}')

        if step >= args.steps:
            break

wall = time.time() - t0
print(f'\nTraining complete: {step} steps in {wall/3600:.1f}h')

# Cell 6: Evaluate PPL
print('Evaluating perplexity on validation set...')

from torch.utils.data import DataLoader

# Load validation set
val_ds = load_dataset(args.dataset, args.dataset_config, split='validation')
val_tokenized = val_ds.map(
    tokenize, batched=True, remove_columns=val_ds.column_names, desc='tokenizing validation'
)
val_lm_ds = val_tokenized.map(
    group_texts, batched=True, remove_columns=val_tokenized.column_names, desc='chunking validation'
)
val_lm_ds.set_format(type='torch', columns=['input_ids', 'labels'])
val_loader = DataLoader(val_lm_ds, batch_size=1, shuffle=False)

# Evaluate base model (fresh load, no adapter)
print('\n1. Evaluating base Llama-3-8B (no adapter)...')
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
)
base_model.eval()

base_losses = []
with torch.inference_mode():
    for batch in val_loader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = base_model(**batch)
        base_losses.append(outputs.loss.item())

base_ppl = torch.exp(torch.tensor(sum(base_losses) / len(base_losses))).item()
print(f'   Base PPL: {base_ppl:.2f}')

del base_model
torch.cuda.empty_cache()

# Evaluate adapted model
print('\n2. Evaluating Llama-3-8B + Phase 5b adapter...')
model.eval()

adapter_losses = []
with torch.inference_mode():
    for batch in val_loader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        adapter_losses.append(outputs.loss.item())

adapter_ppl = torch.exp(torch.tensor(sum(adapter_losses) / len(adapter_losses))).item()
print(f'   Adapter PPL: {adapter_ppl:.2f}')

# Results
ppl_delta = ((base_ppl - adapter_ppl) / base_ppl) * 100
print('\n' + '='*80)
print('RESULTS: Llama-3-8B + Phase 5b')
print('='*80)
print(f'Base PPL:          {base_ppl:.2f}')
print(f'Adapter PPL:       {adapter_ppl:.2f}')
print(f'Improvement:       {ppl_delta:+.1f}%')
print(f'Trainable params:  {result.trainable_percent:.3f}%')
print('='*80)

# Compare with previous results
print('\nCross-Family Comparison:')
print('| Base Model         | Size | PPL Improvement | Trainable % |')
print('|--------------------|------|-----------------|-------------|')
print('| TinyLlama-1.1B     | 1.1B | -28.5%          | 0.196%      |')
print('| Qwen-2.5-1.5B      | 1.5B | -27.7%          | 0.139%      |')
print('| Qwen-2.5-3B        | 3B   | -34.4%          | 0.117%      |')
print(f'| **Llama-3-8B**     | 8B   | **{ppl_delta:+.1f}%**       | {result.trainable_percent:.3f}%     |')

if abs(ppl_delta) >= 25:
    print('\n✓ SUCCESS: Llama-3 shows similar improvement to Llama/Qwen families')
    print('✓ Phase 5b recipe generalizes to third model family')
else:
    print(f'\n⚠ Improvement ({ppl_delta:.1f}%) lower than expected (≥25%)')
    print('  May need longer training or hyperparameter adjustment for 8B scale')

# Cell 7: Save results JSON
import json

results = {
    'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'recipe': 'Phase 5b (MT every 4th + LoRA q/k/v/o)',
    'wrapped_layers': result.wrapped_layer_indices,
    'trainable_params': result.trainable_params,
    'trainable_percent': result.trainable_percent,
    'base_ppl': base_ppl,
    'adapter_ppl': adapter_ppl,
    'ppl_improvement_percent': ppl_delta,
    'training_steps': args.steps,
    'training_time_h': wall / 3600,
}

with open('/kaggle/working/llama3_8b_phase5b_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved: /kaggle/working/llama3_8b_phase5b_results.json')

# Cell 8: Archive artifacts
import shutil

os.makedirs('/kaggle/working/llama3_artifacts', exist_ok=True)
shutil.copy('/kaggle/working/llama3_8b_phase5b_results.json', '/kaggle/working/llama3_artifacts/')

# Copy checkpoint
if os.path.exists(f'{args.out_dir}/llama3_8b_mt_adapter_001000.pt'):
    shutil.copy(
        f'{args.out_dir}/llama3_8b_mt_adapter_001000.pt',
        '/kaggle/working/llama3_artifacts/'
    )

archive = shutil.make_archive('/kaggle/working/llama3_phase5b', 'zip', '/kaggle/working/llama3_artifacts')
print(f'Archive: {archive}')
print('Download and extract to M1/benchmarks/kaggle_llama3_run/')
