# MT-LNN Adapter Recipes

Pre-configured adapter recipes validated across multiple base models. Each recipe is a self-contained function that applies MT adapters ± LoRA with documented hyperparameters.

## Phase 5b Recipe

The **Phase 5b recipe** is the validated configuration that achieved consistent PPL improvements:

| Base Model | PPL Improvement | Trainable % | Validation |
|---|---:|---:|---|
| TinyLlama-1.1B | **-28.5%** | 0.196% | Phase 5 (Kaggle GPU) |
| Qwen-2.5-1.5B | **-27.7%** | 0.139% | Phase 5b (Kaggle GPU) |
| Qwen-2.5-3B | **-34.4%** | 0.117% | Track 1A (Kaggle GPU) |

### Configuration

**MT Adapters:**
- Every 4th decoder layer
- 13 protofilaments (microtubule-inspired parallel processing)
- 5 temporal time scales
- 64-dim MAP gate hidden size
- init_scale = 1e-3 (stable residual connection)

**LoRA (optional):**
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Rank: 8
- Alpha: 16
- Dropout: 0.05

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM
from mt_lnn.recipes import apply_phase5b_recipe

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Apply Phase 5b recipe (MT + LoRA)
result = apply_phase5b_recipe(model)

print(f"Trainable: {result.trainable_percent:.3f}%")
print(f"Layers: {result.wrapped_layer_indices}")

# Model is now ready for training
```

### Command-Line Example

```bash
python examples/apply_phase5b_recipe.py --model Qwen/Qwen2.5-1.5B-Instruct
```

### MT Adapters Only (No LoRA)

```python
from mt_lnn.recipes import apply_mt_only_recipe

result = apply_mt_only_recipe(
    model,
    every=4,
    n_protofilaments=13,
    n_time_scales=5,
)
```

### LoRA Only (No MT)

```python
from mt_lnn.recipes import apply_lora_only_recipe

result = apply_lora_only_recipe(
    model,
    lora_rank=8,
    lora_alpha=16,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

## Ablation Studies

The recipes module makes it trivial to run ablations:

```python
# Ablation 1: Different layer intervals
result_2 = apply_mt_only_recipe(model, every=2)  # Every 2nd layer
result_4 = apply_mt_only_recipe(model, every=4)  # Every 4th layer (Phase 5b)
result_8 = apply_mt_only_recipe(model, every=8)  # Every 8th layer

# Ablation 2: Different LoRA ranks
result_r4 = apply_phase5b_recipe(model, lora_rank=4)
result_r8 = apply_phase5b_recipe(model, lora_rank=8)  # Phase 5b default
result_r16 = apply_phase5b_recipe(model, lora_rank=16)

# Ablation 3: MT vs LoRA vs MT+LoRA
result_mt = apply_mt_only_recipe(model)
result_lora = apply_lora_only_recipe(model)
result_both = apply_phase5b_recipe(model)
```

## Training After Applying Recipe

After applying a recipe, the model is frozen except for the adapter parameters. Train as usual:

```python
from torch.utils.data import DataLoader
from mt_lnn.recipes import apply_phase5b_recipe

# Apply recipe
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
apply_phase5b_recipe(model)

# Setup optimizer (only trainable params)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,
    weight_decay=0.01,
)

# Train
model.train()
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Loading Saved Adapters

To load a checkpoint trained with a recipe:

```python
from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Reconstruct adapter layout from checkpoint
checkpoint = torch.load("llama_mt_adapter_001000.pt")
attach_adapters_from_checkpoint(model, checkpoint)

# Load adapter weights
load_adapter_state(model, "llama_mt_adapter_001000.pt", strict=False)

# Model is ready for inference
```

## API Reference

### `apply_phase5b_recipe(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05, lora_targets=None, verbose=True)`

Apply the validated Phase 5b recipe (MT every 4th layer + LoRA on attention).

**Args:**
- `model`: HuggingFace causal LM (will be frozen)
- `lora_rank`: LoRA rank (set to 0 to disable LoRA)
- `lora_alpha`: LoRA alpha scaling
- `lora_dropout`: LoRA dropout probability
- `lora_targets`: List of module names for LoRA (default: `["q_proj", "k_proj", "v_proj", "o_proj"]`)
- `verbose`: Print parameter counts

**Returns:** `RecipeResult` with wrapped layer indices and parameter counts.

### `apply_mt_only_recipe(model, every=4, n_protofilaments=13, n_time_scales=5, map_hidden_dim=64, init_scale=1e-3, verbose=True)`

Apply MT adapters without LoRA.

**Args:**
- `model`: HuggingFace causal LM
- `every`: Attach MT adapter every N layers
- `n_protofilaments`: Number of parallel protofilaments (default: 13)
- `n_time_scales`: Number of temporal scales
- `map_hidden_dim`: MAP gate hidden dimension
- `init_scale`: Residual initialization scale
- `verbose`: Print parameter counts

**Returns:** `RecipeResult` with wrapped layer indices and parameter counts.

### `apply_lora_only_recipe(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05, lora_targets=None, verbose=True)`

Apply LoRA without MT adapters (vanilla LoRA baseline).

**Args:** Same as `apply_phase5b_recipe` (MT-specific args ignored).

**Returns:** `RecipeResult` with parameter counts.

### `RecipeResult`

Dataclass returned by all recipe functions:

```python
@dataclass
class RecipeResult:
    wrapped_layer_indices: List[int]  # Which layers got MT adapters
    trainable_params: int              # Number of trainable parameters
    total_params: int                  # Total model parameters
    trainable_percent: float           # Trainable / total * 100
    lora_applied: bool                 # Whether LoRA was applied
```

## Cross-Base Reproducibility

The Phase 5b recipe works across different model families:

- **Llama family**: TinyLlama-1.1B-Chat
- **Qwen family**: Qwen-2.5-{0.5B, 1.5B, 3B}-Instruct
- **Should work on**: Mistral, Phi, Llama-3, Gemma (untested)

All models with standard decoder architecture (`model.model.layers`) are supported.

## Next Steps

1. **Run ablations** (Item 3.2): Test different `every` intervals, LoRA ranks, protofilament counts
2. **Test on new bases** (Item 4.2): Apply to Llama-3, Mistral-7B, Phi-3-mini
3. **Document ablation results**: Update this doc with findings

## Implementation Notes

- The recipes module is a thin wrapper around `mt_lnn.llama_adapter`
- All hyperparameters are explicitly documented (no magic numbers)
- The `RecipeResult` dataclass makes it easy to track ablation results
- Recipes freeze the base model automatically
- Compatible with HuggingFace Trainer, raw PyTorch training loops, and Kaggle notebooks
