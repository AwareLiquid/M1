#!/usr/bin/env python3
"""MT-LNN Ablation: Adapter Type - Streamlined"""
import os, sys, subprocess, json, time
from pathlib import Path

print("Setup...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)

REPO, DIR = 'https://github.com/AwareLiquid/M1.git', '/kaggle/working/M1'
if not os.path.exists(DIR):
    subprocess.run(['git', 'clone', '--depth', '1', REPO, DIR], check=True)
os.chdir(DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', 'peft', 'datasets'], check=True)

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from mt_lnn.recipes import apply_mt_only_recipe, apply_lora_only_recipe, apply_phase5b_recipe

print("Ablation: MT vs LoRA vs Both (Qwen-1.5B, 200 steps each)")

MODEL, STEPS, BATCH, SEQ_LEN, GRAD_ACCUM = 'Qwen/Qwen2.5-1.5B-Instruct', 200, 1, 384, 8
configs = [
    ('mt_only', lambda m: apply_mt_only_recipe(m, every=4)),
    ('lora_only', lambda m: apply_lora_only_recipe(m, lora_rank=8)),
    ('mt_plus_lora', lambda m: apply_phase5b_recipe(m, lora_rank=8)),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
tokenized = ds.map(lambda b: tokenizer([t for t in b['text'] if t], add_special_tokens=False) if any(b['text']) else {'input_ids': []}, batched=True, remove_columns=ds.column_names)

def group(ex):
    ids = sum([r + [tokenizer.eos_token_id] for r in ex['input_ids']], [])
    total = (len(ids) // SEQ_LEN) * SEQ_LEN
    chunks = [ids[i:i+SEQ_LEN] for i in range(0, total, SEQ_LEN)]
    return {'input_ids': chunks, 'labels': chunks}

lm_ds = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)
lm_ds.set_format('torch', columns=['input_ids', 'labels'])
loader = DataLoader(lm_ds, batch_size=BATCH, shuffle=True, drop_last=True)

results = []
for name, apply_fn in configs:
    print(f"\n=== {name} ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.config.use_cache, model.gradient_checkpointing_enable = False, model.gradient_checkpointing_enable()
    res = apply_fn(model)
    model.to('cuda').train()
    print(f"Trainable: {res.trainable_percent:.3f}%")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    step, losses, t0 = 0, [], time.time()
    
    while step < STEPS:
        for batch in loader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            losses.append(loss.item() * GRAD_ACCUM)

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 20 == 0: print(f"  {step}/{STEPS} | ppl {torch.exp(torch.tensor(sum(losses[-20:])/20)):.2f}")
            if step >= STEPS: break

    final = sum(losses[-20:])/20
    results.append({'name': name, 'ppl': torch.exp(torch.tensor(final)).item(), 'time': time.time()-t0, 'trainable': res.trainable_percent})
    del model, optimizer
    torch.cuda.empty_cache()

Path('/kaggle/working/ablation_results').mkdir(exist_ok=True)
with open('/kaggle/working/ablation_results/adapter_type_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nRESULTS:")
for r in sorted(results, key=lambda x: x['ppl']):
    print(f"{r['name']:<15} PPL {r['ppl']:6.2f} | {r['trainable']:5.2f}% | {r['time']/60:4.1f}min")
