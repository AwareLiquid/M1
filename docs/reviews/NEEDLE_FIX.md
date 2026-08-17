# Needle-in-a-Haystack Test Fix

## Problem

The original `bench_llama_mt_needle.py` produced **0.0 accuracy** on all instruct-tuned models (TinyLlama-1.1B-Chat, Qwen-1.5B-Instruct, Qwen-3B-Instruct) because it used raw prompt concatenation without proper chat template formatting.

### Original Format (Broken)

```python
FILLER = " The archive contains ordinary notes..."
QUESTION = "\nQuestion: What is the secret passcode? Answer with only the digits.\nAnswer:"

# Direct concatenation without chat template
prompt = filler_before + needle + filler_after + question
```

This raw format doesn't match the chat template expected by instruct-tuned models like Qwen-2.5:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

## Solution

Created `bench_needle_chat_template.py` that uses `tokenizer.apply_chat_template()` to properly format prompts:

```python
def make_prompt_with_chat_template(tokenizer, context_len, depth, code):
    needle = f"Important memory: the secret passcode is {code}. Remember this exact passcode."
    question = "Question: What is the secret passcode? Answer with only the digits."
    
    # Construct user message with needle embedded in filler
    user_message = filler_before + "\n\n" + needle + "\n\n" + filler_after + "\n\n" + question
    
    messages = [{"role": "user", "content": user_message}]
    
    # Apply model-specific chat template
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    return prompt_ids
```

## Results

**Before fix**: 0.0 accuracy on all models  
**After fix**: 1.0 accuracy on Qwen-0.5B-Instruct (tested 2025-05-30)

### Qwen-0.5B-Instruct Baseline (CPU)

| Context | Depth | Exact | Contains | Tok/s | Seconds |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.10 | 1.000 | 1.000 | 290 | 5.2 |
| 512 | 0.50 | 1.000 | 1.000 | 290 | 5.2 |
| 512 | 0.90 | 1.000 | 1.000 | 251 | 6.0 |
| 1024 | 0.10 | 1.000 | 1.000 | 302 | 10.1 |
| 1024 | 0.50 | 1.000 | 1.000 | 326 | 9.3 |
| 1024 | 0.90 | 1.000 | 1.000 | 316 | 9.6 |

**Perfect retrieval** at all depths (0.1, 0.5, 0.9) and context lengths (512, 1024).

## Usage

```bash
# Test base model
python bench_needle_chat_template.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --context_lengths 512 1024 2048 \
    --depths 0.1 0.5 0.9 \
    --samples 5 \
    --out_json artifacts/needle_base.json

# Test with MT adapter
python bench_needle_chat_template.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapters benchmarks/kaggle_qwen_run/llama_mt_adapter_001000.pt \
    --context_lengths 512 1024 2048 \
    --depths 0.1 0.5 0.9 \
    --samples 5 \
    --out_json artifacts/needle_adapter.json
```

## Next Steps

- [ ] Run on Qwen-1.5B-Instruct with Phase 5b adapter to validate MT adapter performance
- [ ] Run on Qwen-3B-Instruct with Phase 5b adapter to test longer contexts (4k, 8k)
- [ ] Compare base vs +adapter accuracy to see if MT architecture improves retrieval
- [ ] Add to Kaggle notebook for cloud validation

## Technical Notes

- Uses `tokenizer.apply_chat_template()` with `add_generation_prompt=True`
- Includes fallback for models without native chat template support
- Properly accounts for template overhead when calculating token budgets
- Maintains same evaluation logic (exact match, contains, tokens/sec)
