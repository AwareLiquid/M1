"""ARR mixer streaming-state parity (O-series analogue of
test_streaming_state.py). Chunked forward with mixer state carried must
match the full-sequence forward; without state it must NOT."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.arr import (convert_to_arr, probe_return_convention,
                        reset_mixer_streams, set_mixer_streaming)

CHUNKS = [7, 9, 15, 1]
T = sum(CHUNKS)


def test_arr_streaming_parity():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=6, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg)
    convert_to_arr(m, d_proto=16, proj_rank=16, fast_weight_dim=8)
    probe_return_convention(m)
    m.eval()
    ids = torch.randint(0, 128, (1, T), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        full = m(input_ids=ids).logits

    def chunked():
        outs, pos = [], 0
        with torch.no_grad():
            for c in CHUNKS:
                outs.append(m(input_ids=ids[:, pos: pos + c]).logits)
                pos += c
        return torch.cat(outs, dim=1)

    # Control: stateless chunking must diverge (mixers ARE the token mixing).
    stale = chunked()
    assert not torch.allclose(full, stale, rtol=1e-3, atol=1e-4), \
        "stateless chunking matched full forward — mixers inert?"

    # Streaming: exact parity.
    n = set_mixer_streaming(m, True)
    assert n == 6
    reset_mixer_streams(m)
    streamed = chunked()
    diff = (full - streamed).abs().max().item()
    assert torch.allclose(full, streamed, rtol=1e-4, atol=1e-5), \
        f"ARR streaming parity failed: max|diff|={diff:.2e}"

    # Train-through keeps graph.
    set_mixer_streaming(m, True, train_through=True)
    m.train()
    m(input_ids=ids[:, :8])
    from mt_lnn.arr import MTRecurrentMixer
    mix = next(mm for mm in m.modules() if isinstance(mm, MTRecurrentMixer))
    assert mix._stream_h is not None and mix._stream_h.grad_fn is not None
    print(f"[arr] streaming parity (max|diff| {diff:.1e}) + train-through  OK")


if __name__ == "__main__":
    test_arr_streaming_parity()
    print("all arr streaming tests passed")
