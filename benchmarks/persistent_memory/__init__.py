"""PMB — Persistent Memory Benchmark v0.

Cross-session persistent-memory benchmark (see docs/specs/PMB_SPEC.md).
Three task families (T1 retention / T2 streaming update / T3 forgetting
curve), a unified MemorySystem protocol with mandatory disk serialization
between sessions, and a CLI runner reporting the mandatory three columns:
recall, state_bytes, answer_latency_ms.
"""
