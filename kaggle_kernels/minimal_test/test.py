#!/usr/bin/env python3
"""Minimal test to isolate the error"""
import os, sys, subprocess

print("=== Step 1: Install torch ===")
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True, timeout=300)
    print("? Torch installed")
except Exception as e:
    print(f"? Torch install failed: {e}")
    sys.exit(1)

print("\n=== Step 2: Clone M1 ===")
REPO, DIR = 'https://github.com/AwareLiquid/M1.git', '/kaggle/working/M1'
try:
    if not os.path.exists(DIR):
        subprocess.run(['git', 'clone', '--depth', '1', REPO, DIR], check=True, timeout=300)
    os.chdir(DIR)
    print("? M1 cloned")
except Exception as e:
    print(f"? Clone failed: {e}")
    sys.exit(1)

print("\n=== Step 3: Install requirements ===")
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', 'peft', 'datasets'], check=True, timeout=600)
    print("? Requirements installed")
except Exception as e:
    print(f"? Requirements install failed: {e}")
    sys.exit(1)

print("\n=== Step 4: Import packages ===")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"? Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"? Import failed: {e}")
    sys.exit(1)

print("\n=== Step 5: Import MT-LNN recipes ===")
try:
    from mt_lnn.recipes import apply_mt_only_recipe
    print("? MT-LNN recipes imported")
except Exception as e:
    print(f"? Recipe import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 6: Load Qwen-0.5B (smaller model) ===")
try:
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.float16)
    print(f"? Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"? Model load failed: {e}")
    sys.exit(1)

print("\n=== Step 7: Apply MT-only recipe ===")
try:
    result = apply_mt_only_recipe(model, every=4, verbose=True)
    print(f"? Recipe applied: {result.trainable_percent:.3f}% trainable")
except Exception as e:
    print(f"? Recipe failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 8: Move to GPU and test forward pass ===")
try:
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    inputs = tokenizer("Test", return_tensors='pt').to('cuda')
    with torch.inference_mode():
        outputs = model(**inputs)
    print(f"? Forward pass works, loss: {outputs.loss if hasattr(outputs, 'loss') else 'N/A'}")
except Exception as e:
    print(f"? Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("? ALL TESTS PASSED")
print("The issue is likely in the training loop, not setup/recipes")
print("="*60)
