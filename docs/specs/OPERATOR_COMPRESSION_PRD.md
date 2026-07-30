# PRD: Operator Compression for MT-LNN

## Goal

Turn MT-LNN from a standard autoregressive decoder into a hybrid operator-compression system:

- Compress target extraction into one global-state projection.
- Compress long context into recurrent `h_prev` state instead of replayed tokens.
- Compress multi-scale computation by activating only the needed tau channels.
- Optionally skip upstream resonance work by selecting top-k tau scales before
  the per-scale projection and recurrent scan.

The product claim must stay precise:

> MT-LNN can expose compressed inference paths that reduce token-level generation, replay, and cache growth for tasks whose structure supports compression.

## Non-Goals

- Do not claim general language reasoning is solved by function compression alone.
- Do not remove normal autoregressive generation.
- Do not persist KV cache as long-term memory; KV is position-tied token history.
- Do not enable compute skipping by default until benchmarked.
- Do not present the optional sparse resonance path as a custom CUDA kernel;
  the current implementation is a conservative PyTorch top-k path over tau scales.

## Architecture

```text
                      +-----------------------------+
                      | input tokens / new token     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Embedding + RoPE             |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
   +-----------------+        +-----------------+        +-----------------+
   | MT Attention    |        | MT-LNN h_prev   |        | GWTB/Coherence  |
   | KV cache path   |        | recurrent path  |        | global state    |
   +--------+--------+        +--------+--------+        +--------+--------+
            |                          |                          |
            +--------------+-----------+--------------+-----------+
                           |                          |
                        v                          v
                +---------------------+    +---------------------+
                | LM head             |    | Direct target head   |
                | autoregressive      |    | one-shot targets     |
                +---------------------+    +---------------------+
```

## Inference Modes

### 1. Normal Autoregressive Mode

```text
prompt tokens -> prefill KV+h_prev -> token-by-token decode
```

Use when exact causal attention over the prompt is needed.

### 2. Direct Target Extraction

```text
prefix -> final global state -> target slots -> target_logits
```

Use when the task has fixed output slots, such as Selective Copy or structured extraction.

### 3. State-Only Streaming

```text
new token -> h_prev update -> logits
          -> drop KV history
          -> persist h_prev only
```

Use when the goal is compressed continuity across many turns without replaying history.

Important tradeoff:
- Keeps recurrent memory.
- Drops exact attention over old tokens.
- Best treated as compressed memory, not lossless transcript recall.

## Cache Contract

`ModelCacheStruct` carries:

- Per layer attention KV: optional, grows with tokens in normal mode.
- Per layer `h_prev`: recurrent state, constant-size per layer.
- Per-block GWTB KV: optional, grows with tokens in normal mode.
- Top-level GWTB/coherence KV: optional, grows with tokens in normal mode.
- `token_count`: metadata for absolute positions without storing token history.

State-only mode returns a cache where:

- KV slots are `None`.
- `h_prev` slots are preserved.
- `token_count` advances.

## Development Milestones

1. Day1: direct target extraction.
2. Day2: state-only streaming helper.
3. Day3: persistent session workflow.
4. Day4: stress benchmark.
5. Day5: dynamic tau-scale gates.
6. Day6: optional compute skipping and demo integration.
7. Day7: benchmark report and README updates.

## Evaluation

Metrics:

- Direct extraction token accuracy and sequence exact match.
- Per-token latency in normal KV mode vs state-only mode.
- Cache bytes as sequence length grows.
- Divergence between full replay, KV cache, and state-only recurrent mode.
- Tau gate activity distribution once channel gating lands.
- Sparse resonance selected-scale ratio and output divergence when enabled.
- Structured JSONL metric events for benchmark modes, cache bytes, and gate diagnostics.

## Risks

- SQLite-backed `SessionMemory` is a local demo/session mechanism, not a
  distributed high-QPS state service. Production serving needs an explicit
  storage adapter boundary plus connection management and backpressure.
- `requirements.txt` keeps flexible lower bounds for experimentation;
  `requirements.lock` should be used for reproducible benchmark reruns.

- State-only mode is not exact long-context recall.
- Direct heads require task-specific supervision.
- Dynamic channel skipping can destabilize training if gates close too early.
- Sparse resonance top-k selection is chunk-dependent; when enabled it is a
  speed/compute experiment, not the exact KV-parity path.
- Marketing language must separate measured benchmark results from architecture hypotheses.
