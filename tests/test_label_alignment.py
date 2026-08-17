"""
tests/test_label_alignment.py — guards the causal-LM label convention.

A subtle, catastrophic bug once shipped here: ``train.BinDataset`` pre-shifted
its labels (``y = chunk[1:]``) while ``MTLNNModel.forward`` *also* shifts
internally (HF style: ``shift_logits = logits[:, :-1]`` vs
``shift_labels = labels[:, 1:]``). The double-shift turned the training target
into a *skip-one* objective — predict ``chunk[i+2]`` from ``chunk[:i]`` — which
looked healthy in train loss (PPL ~20) but destroyed autoregressive generation
(true next-token PPL ~800+).

These tests pin the contract so it can never silently regress:

  1. ``BinDataset`` / ``DummyDataset`` return labels ALIGNED with inputs
     (``labels == x``), NOT pre-shifted.
  2. The model's internal ``labels`` loss equals the next-token cross-entropy
     computed from ``logits`` (i.e. the loss the autoregressive decoder
     actually minimises), to within float tolerance.
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
import train as T

D = 104


def _model(max_seq_len=64):
    cfg = MTLNNConfig(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


# --- dataset label convention --------------------------------------------

def test_bindataset_labels_aligned_not_preshifted(tmp_path):
    # Build a tiny uint16 token stream on disk.
    p = tmp_path / "toy.bin"
    np.arange(200, dtype=np.uint16).tofile(p)
    ds = T.BinDataset(str(p), seq_len=16)
    x, y = ds[0]
    # labels must be aligned with inputs (model shifts internally), NOT chunk[1:]
    assert torch.equal(x, y), "BinDataset labels must equal inputs (no pre-shift)"
    # and must NOT be the next-token shift (the old bug)
    assert not torch.equal(y[:-1], x[1:]) or torch.equal(x, y)


def test_dummydataset_labels_aligned():
    ds = T.DummyDataset(vocab_size=64, seq_len=16, n_samples=4)
    x, y = ds[0]
    assert torch.equal(x, y)


# --- the core invariant: internal loss == autoregressive next-token loss ---

def test_model_labels_loss_equals_autoregressive_next_token():
    torch.manual_seed(0)
    m = _model()
    seq = torch.randint(0, 64, (2, 33))           # (B, seq_len+1)
    x = seq[:, :-1]                                # inputs
    with torch.no_grad():
        logits = m(x)["logits"]                    # (B, T, V)
        internal = m(x, labels=x.clone())["loss"]  # HF-aligned labels
    # Autoregressive next-token CE: logit_i (saw x[:i]) predicts x[i+1].
    nt = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        x[:, 1:].reshape(-1),
    )
    assert torch.allclose(internal, nt, atol=1e-5), (
        f"internal labels-loss {internal.item():.5f} != next-token CE "
        f"{nt.item():.5f} — label alignment regressed"
    )


def test_preshifted_labels_would_diverge():
    # Sanity: the OLD convention (labels = chunk[1:]) must give a DIFFERENT
    # loss — proving the test above is actually sensitive to the bug.
    torch.manual_seed(0)
    m = _model()
    seq = torch.randint(0, 64, (2, 33))
    x, y_buggy = seq[:, :-1], seq[:, 1:]
    with torch.no_grad():
        logits = m(x)["logits"]
        buggy = m(x, labels=y_buggy)["loss"]       # double-shift (skip-one)
    nt = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        x[:, 1:].reshape(-1),
    )
    assert not torch.allclose(buggy, nt, atol=1e-4), (
        "pre-shifted labels happened to match next-token CE — test is not "
        "sensitive to the double-shift bug"
    )


if __name__ == "__main__":
    import tempfile, pathlib, traceback
    tmp = pathlib.Path(tempfile.mkdtemp())
    tests = [
        ("test_bindataset_labels_aligned_not_preshifted", lambda: test_bindataset_labels_aligned_not_preshifted(tmp)),
        ("test_dummydataset_labels_aligned", test_dummydataset_labels_aligned),
        ("test_model_labels_loss_equals_autoregressive_next_token", test_model_labels_loss_equals_autoregressive_next_token),
        ("test_preshifted_labels_would_diverge", test_preshifted_labels_would_diverge),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"[ok] {name}")
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            raise
