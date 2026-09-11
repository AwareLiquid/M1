# Experimental development moved to M2

The active research repository is [AwareLiquid/M2](https://github.com/AwareLiquid/M2).
Its initial code import is commit `47bc830b710c1960453ac6bdc0d8dedf8dec4c89`.

## Provenance

The import uses the preserved `everest-an/M1` PR #46 branch at
`1e0114d10cc9b4edb3b60cf3aec428769a03b698`, including the shared runtime
required by the experiments. That PR was closed without merging. This is not
a claim that the organizational M1 checkout has the same source revision.

M2 contains recurrent core/stack experiments, GWT workspace, coherence,
world-model and predictive-coding modules, top-down signals, rhythm and
plasticity components, plus their runtime dependencies and selected tests.
Migration validation: 156 tests passed and the GPU reasoning-depth smoke
completed training and evaluation. These are execution checks, not new
capability results.

## Compatibility and ownership

M1 retains historical modules and checkpoint paths for compatibility.
No default graph or trained weights changed in this documentation update.
New experimental development belongs in M2. Removing compatibility copies
from M1 requires a separate caller/checkpoint migration and regression check.

Both repositories currently expose `mt_lnn`; use separate Python environments.
The code was imported as a fixed snapshot, not a dynamically shared dependency.

The former `AwareLiquid/AwareLiquid-M2` repository is now
[AwareLiquid-RAG](https://github.com/AwareLiquid/AwareLiquid-RAG), an independent
document-QA adapter. Its results must not be attributed to the M2 architecture.
