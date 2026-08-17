"""State-only streaming helpers for MT-LNN.

These helpers expose the recurrent `h_prev` path as a first-class inference
mode. Normal generation can still keep full KV history; state-only streaming
drops token-position KV tensors after each step and carries only the constant
size recurrent state.
"""

from typing import Optional, Tuple

import torch

from .model import ModelCacheStruct, MTLNNModel


def _as_single_token(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.dim() != 2 or input_ids.shape[1] != 1:
        raise ValueError(
            "streaming_inference expects one token per batch item; "
            f"got shape {tuple(input_ids.shape)}"
        )
    return input_ids


@torch.no_grad()
def streaming_inference(
    model: MTLNNModel,
    new_token: torch.Tensor,
    cache: Optional[ModelCacheStruct] = None,
    *,
    token_count: Optional[int] = None,
    state_only: bool = True,
    use_lnn_recurrence: bool = True,
) -> Tuple[torch.Tensor, ModelCacheStruct]:
    """Run one streaming inference step.

    Parameters
    ----------
    model:
        MT-LNN model in eval or train mode.
    new_token:
        Tensor shaped ``(B, 1)`` or ``(B,)``.
    cache:
        Previous cache. In ``state_only`` mode, only its recurrent states are
        used and returned.
    token_count:
        Absolute number of tokens previously consumed. Defaults to
        ``cache.token_count`` or zero.
    state_only:
        If True, return a recurrent-only cache with no historical KV tensors.
    use_lnn_recurrence:
        Forwarded to the model.

    Returns
    -------
    logits, cache:
        ``logits`` is ``(B, 1, vocab_size)``. ``cache`` is the updated cache.
    """
    new_token = _as_single_token(new_token)

    if token_count is None:
        token_count = cache.token_count if cache is not None else 0

    seed_cache = cache.recurrent_only() if (cache is not None and state_only) else cache
    # State-only mode has no historical KV positions. Keep absolute-time
    # features bounded by the model's finite RoPE/mask tables.
    position_offset = token_count % model.config.max_seq_len if state_only else token_count

    out = model(
        new_token,
        cache=seed_cache,
        use_cache=True,
        position_offset=position_offset,
        use_lnn_recurrence=use_lnn_recurrence,
    )

    next_cache = out["cache"]
    next_cache.token_count = token_count + new_token.shape[1]
    if state_only:
        next_cache = next_cache.recurrent_only()
        next_cache.token_count = token_count + new_token.shape[1]
    return out["logits"], next_cache


@torch.no_grad()
def prefill_state_only(
    model: MTLNNModel,
    input_ids: torch.Tensor,
    cache: Optional[ModelCacheStruct] = None,
    *,
    token_count: Optional[int] = None,
    use_lnn_recurrence: bool = True,
) -> Tuple[torch.Tensor, ModelCacheStruct]:
    """Consume a prompt token by token while keeping only recurrent state."""
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, T), got {tuple(input_ids.shape)}")

    logits = None
    cur_cache = cache
    cur_count = cache.token_count if token_count is None and cache is not None else (token_count or 0)
    for t in range(input_ids.shape[1]):
        logits, cur_cache = streaming_inference(
            model,
            input_ids[:, t:t + 1],
            cur_cache,
            token_count=cur_count,
            state_only=True,
            use_lnn_recurrence=use_lnn_recurrence,
        )
        cur_count = cur_cache.token_count
    return logits, cur_cache

