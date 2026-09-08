# M1 / M2 architecture boundary

M1 is the reproducible product and publication trunk. M2 is the opt-in
research surface for mechanisms that still need causal ablation.

## M1 stable

Create the stable graph with:

```python
from mt_lnn import m1_stable_config

config = m1_stable_config()
```

The profile enables the confirmed modern language trunk
(`ffn_swiglu`, `qk_norm`, `scaled_residual_init`) and removes GWTB, global
coherence, latent-depth loops, predictive/world-model heads, rhythm,
top-down modulation and core fast weights. `MTLNNConfig()` remains the
historical checkpoint-compatible configuration.

The modern trunk's 20K result is supported, but its current 195.1M form is
35% larger than the 144.1M Transformer anchor. It must not be described as a
matched-parameter win until the planned approximately 148M run lands.

## M2 research

Import research profiles from the dedicated namespace and enable only the
mechanism under test:

```python
from mt_lnn.m2 import M2Experiment, m2_research_config

config = m2_research_config((M2Experiment.SELECTIVE_STATE,))
```

Multiple experiments may be composed explicitly, but single-mechanism runs
are the default scientific protocol:

```python
config = m2_research_config(
    (
        M2Experiment.WORKSPACE,
        M2Experiment.FAST_WEIGHT_MEMORY,
    )
)
```

Available research families are selective state transitions, core fast
weights, latent core/stack recurrence, workspace and competitive workspace,
top-down modulation, predictive coding, world models, global rhythm, global
coherence and Hebbian plasticity. The workspace preset automatically enables
its required GWTB parent.

## Graduation rule

An M2 mechanism moves into M1 only after all of the following hold:

1. The primary metric improves under a pre-registered protocol.
2. At least three seeds reproduce the direction.
3. A direct ablation attributes the gain to that mechanism.
4. Parameter, training-FLOP, inference-latency and memory costs are reported.
5. The result survives a strong contemporary baseline and a matched-budget
   comparison.

Moving a mechanism into M1 requires a new explicit profile change. It never
happens by changing the historical `MTLNNConfig()` defaults silently.
