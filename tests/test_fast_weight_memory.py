"""tests/test_fast_weight_memory.py -- FastWeightMemory associative recall.

The Hebbian co-activation REGULARIZER is inert at small scale (gradient ~1e-12,
drowned by CE). FastWeightMemory is the architectural answer: a forward-pass,
content-addressable memory trained directly by the task loss, so associative
recall is learnable at 0.5B scale -- the principled fix for the served adapter's
0% needle-in-haystack result.

Coverage:
  1. unit -- output shape, strict causality, gradient reaches every projection.
  2. disabled-by-default -- use_fast_weight=False leaves the served graph
     byte-identical (no fast_weight module, output unchanged).
  3. associative recall (needle) -- a small end-to-end training on a key->value
     recall task: with fast weights ON the model learns to recall a value seen
     once, far earlier in the sequence; the memoryless control cannot match it.
"""

import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.llama_adapter import FastWeightMemory, attach_mt_adapters


# --------------------------------------------------------------------------- #
# Minimal attention-free backbone: the ONLY way information crosses positions  #
# is through the adapter, so a recall win is attributable to the adapter.      #
# --------------------------------------------------------------------------- #
class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.proj(hidden_states),)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size=64, n_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyDecoderLayer(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


# --------------------------------------------------------------------------- #
# 1. Unit: shape, causality, gradient                                          #
# --------------------------------------------------------------------------- #
def test_fast_weight_shape_and_grad():
    torch.manual_seed(0)
    fw = FastWeightMemory(d_model=32, d_mem=16, n_heads=2)
    x = torch.randn(3, 7, 32, requires_grad=True)
    out, (Fmat, zvec) = fw(x)
    assert out.shape == (3, 7, 32)
    assert Fmat.shape == (3, 2, 16, 16)
    assert zvec.shape == (3, 2, 16)
    out.square().mean().backward()
    # gradient must reach every projection AND the learnable decay
    for name, p in fw.named_parameters():
        assert p.grad is not None and torch.any(p.grad != 0), f"no grad on {name}"


def test_fast_weight_is_causal():
    """Output at position t must not depend on inputs at positions > t.
    Perturbing a FUTURE token must leave earlier outputs bit-identical."""
    torch.manual_seed(0)
    fw = FastWeightMemory(d_model=24, d_mem=12, n_heads=1)
    fw.eval()
    x = torch.randn(2, 10, 24)
    with torch.no_grad():
        out_a, _ = fw(x)
        x2 = x.clone()
        x2[:, 6:] += torch.randn_like(x2[:, 6:])  # change tokens 6..9 only
        out_b, _ = fw(x2)
    # positions 0..5 must be unchanged; 6.. may change
    assert torch.allclose(out_a[:, :6], out_b[:, :6], atol=1e-6), \
        "future tokens leaked into past outputs -- not causal"
    assert not torch.allclose(out_a[:, 6:], out_b[:, 6:], atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Disabled by default -> served graph untouched                             #
# --------------------------------------------------------------------------- #
def test_fast_weight_off_by_default():
    torch.manual_seed(0)
    model = TinyBackbone()
    attach_mt_adapters(model, every=2, n_protofilaments=4, n_time_scales=2,
                       map_hidden_dim=8, init_scale=1e-2)
    # no adapter carries a fast_weight module
    from mt_lnn.llama_adapter import MTResidualAdapter
    adapters = [m for m in model.modules() if isinstance(m, MTResidualAdapter)]
    assert adapters and all(a.fast_weight is None for a in adapters)


# --------------------------------------------------------------------------- #
# 3. Associative recall (needle): the falsifiable mechanism claim              #
# --------------------------------------------------------------------------- #
class TokenRecallModel(nn.Module):
    """emb -> attention-free backbone (+MT adapter) -> readout. Cross-position
    information can ONLY flow through the adapter, so recall is attributable."""

    def __init__(self, vocab, hidden, use_fast_weight):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.backbone = TinyBackbone(hidden_size=hidden, n_layers=4)
        attach_mt_adapters(
            self.backbone, every=2, n_protofilaments=4, n_time_scales=2,
            map_hidden_dim=16, init_scale=0.1,
            use_fast_weight=use_fast_weight, fast_weight_dim=32, fast_weight_heads=2,
        )
        # un-freeze emb/readout (attach_mt_adapters froze the backbone base layers)
        self.readout = nn.Linear(hidden, vocab)
        for p in self.emb.parameters():
            p.requires_grad = True
        for p in self.readout.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.readout(self.backbone(self.emb(x)))


# vocab layout (disjoint ranges so key/value never collide with filler)
N_KEY, N_VAL, N_FILL = 12, 12, 24
KEY0, VAL0, FILL0 = 0, N_KEY, N_KEY + N_VAL
VOCAB = N_KEY + N_VAL + N_FILL
T = 32


def _make_recall_batch(B, seed):
    """Each sample: a [key, value] pair sits once at a random early position
    amid filler; the LAST token re-presents the key. Target = the value, to be
    predicted at the final position from memory alone."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(FILL0, VOCAB, (B, T), generator=g)          # filler everywhere
    keys = torch.randint(KEY0, KEY0 + N_KEY, (B,), generator=g)
    vals = torch.randint(VAL0, VAL0 + N_VAL, (B,), generator=g)
    pos = torch.randint(0, T - 4, (B,), generator=g)              # early needle slot
    for b in range(B):
        x[b, pos[b]] = keys[b]
        x[b, pos[b] + 1] = vals[b]
    x[:, -1] = keys                                              # query at the end
    return x, vals


def _train_recall(use_fast_weight, steps=400, hidden=64, batch=64, seed=0):
    torch.manual_seed(seed)
    model = TokenRecallModel(VOCAB, hidden, use_fast_weight)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    for step in range(steps):
        x, tgt = _make_recall_batch(batch, seed=1000 + step)
        logits = model(x)[:, -1, :]                             # final-position logits
        loss = lossf(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    # held-out accuracy on unseen seeds
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ev in range(5):
            x, tgt = _make_recall_batch(128, seed=90000 + ev)
            pred = model(x)[:, -1, :].argmax(-1)
            correct += int((pred == tgt).sum())
            total += tgt.numel()
    return correct / total


def test_fast_weight_enables_associative_recall():
    """Falsifiable mechanism claim: with fast weights ON, the model learns to
    recall a value seen once far earlier (needle); the memoryless control (LTC
    adapter only, no fast weights) does substantially worse. Chance ~= 1/12."""
    acc_fw = _train_recall(use_fast_weight=True)
    acc_ctrl = _train_recall(use_fast_weight=False)
    print(f"\n[recall] fast-weight acc={acc_fw:.3f}  control acc={acc_ctrl:.3f}  "
          f"(chance ~{1.0 / N_VAL:.3f})")
    # primary: the mechanism genuinely recalls, far above chance
    assert acc_fw > 0.55, (
        f"fast-weight recall {acc_fw:.3f} not above chance -- mechanism failed"
    )
    # secondary: fast weights help relative to the memoryless control
    assert acc_fw > acc_ctrl + 0.15, (
        f"fast-weight ({acc_fw:.3f}) did not beat control ({acc_ctrl:.3f}) by a "
        f"clear margin -- recall not attributable to the fast-weight memory"
    )


if __name__ == "__main__":
    for fn in (test_fast_weight_shape_and_grad, test_fast_weight_is_causal,
               test_fast_weight_off_by_default,
               test_fast_weight_enables_associative_recall):
        fn()
        print(f"PASS {fn.__name__}")
