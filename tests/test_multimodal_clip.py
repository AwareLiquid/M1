"""
tests/test_multimodal_clip.py — CLIP vision-tower bridge + the multimodal
serving endpoint.

The real CLIP tower needs `transformers` and a one-time weight download, so the
core tests use a tiny **stub tower** (exposes `.hidden_size` + `.encode`) that
stands in for CLIP. This pins the wiring contract offline:

  • CLIPModalityEncoder(tower=stub) → (B, N, d_model) tokens.
  • The projector is trainable; gradient reaches its weights.
  • CLIP-style tokens fuse and flow through MTLNNModel.forward(inputs_embeds=...).
  • The FastAPI /v1/multimodal/completions endpoint: 503 when disabled, and a
    full image→text round-trip when an encoder is injected.

A separate, network-guarded test exercises the genuine CLIP download path and
skips cleanly when transformers/weights are unavailable.
"""

import base64
import io
import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.multimodal import CLIPModalityEncoder, fuse

D = 104
CLIP_HIDDEN = 32
N_PATCH = 4


class _StubTower(torch.nn.Module):
    """Stand-in for a CLIP vision tower: turns images into patch features."""

    hidden_size = CLIP_HIDDEN

    def encode(self, images):
        n = len(images) if not isinstance(images, torch.Tensor) else images.shape[0]
        return torch.randn(n, N_PATCH, self.hidden_size)


def _model(max_seq_len=64):
    cfg = MTLNNConfig(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


# --- encoder bridge -------------------------------------------------------

def test_clip_modality_encoder_shape_with_stub_tower():
    enc = CLIPModalityEncoder(d_model=D, tower=_StubTower())
    out = enc([object(), object()])           # 2 "images"
    assert out.shape == (2, N_PATCH, D)


def test_clip_modality_encoder_requires_hidden_size():
    class _NoHidden(torch.nn.Module):
        def encode(self, images):            # pragma: no cover - never reached
            return torch.zeros(1, 1, 1)
    with pytest.raises(ValueError):
        CLIPModalityEncoder(d_model=D, tower=_NoHidden())


def test_clip_projector_is_trainable():
    enc = CLIPModalityEncoder(d_model=D, tower=_StubTower())
    enc.train()
    out = enc([object()])
    out.sum().backward()
    w = enc.projector.proj.weight
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_clip_tokens_fuse_and_forward():
    torch.manual_seed(0)
    m = _model()
    enc = CLIPModalityEncoder(d_model=D, tower=_StubTower())
    ids = torch.randint(0, 64, (1, 5))
    fused = fuse(enc([object()]), m.embed_tokens(ids))   # (1, N_PATCH+5, D)
    with torch.no_grad():
        out = m(inputs_embeds=fused, use_cache=True)
    assert out["logits"].shape == (1, N_PATCH + 5, 64)
    assert torch.isfinite(out["logits"]).all()
    assert out["cache"].token_count == N_PATCH + 5


# --- serving endpoint -----------------------------------------------------

def _png_b64(color=(255, 0, 0), size=(8, 8)) -> str:
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _client(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("SMALL", "1")
    from fastapi.testclient import TestClient
    import serve.server as srv
    return srv, TestClient(srv.app)


def test_multimodal_endpoint_disabled_by_default(monkeypatch):
    srv, client = _client(monkeypatch)
    with client:
        assert client.get("/health").json()["multimodal"] is False
        r = client.post("/v1/multimodal/completions",
                        json={"prompt": "hi", "images": [_png_b64()]})
        assert r.status_code == 503
        assert "ENABLE_MULTIMODAL" in r.json()["detail"]


def test_multimodal_endpoint_roundtrip_with_injected_encoder(monkeypatch):
    srv, client = _client(monkeypatch)
    with client:
        d_model = srv._STATE["model"].config.d_model
        srv._STATE["mm_encoder"] = CLIPModalityEncoder(
            d_model=d_model, tower=_StubTower()
        ).eval()
        srv._STATE["mm_ready"] = True
        srv._STATE["mm_model"] = "stub"

        r = client.post("/v1/multimodal/completions",
                        json={"prompt": "look:", "images": [_png_b64()],
                              "max_new_tokens": 5, "do_sample": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_images"] == 1
        assert body["n_image_tokens"] == N_PATCH
        assert body["n_new_tokens"] == 5
        assert isinstance(body["text"], str)
        assert client.get("/v1/model").json()["multimodal"] is True


def test_multimodal_endpoint_rejects_bad_base64(monkeypatch):
    srv, client = _client(monkeypatch)
    with client:
        srv._STATE["mm_encoder"] = CLIPModalityEncoder(
            d_model=srv._STATE["model"].config.d_model, tower=_StubTower()
        ).eval()
        srv._STATE["mm_ready"] = True
        r = client.post("/v1/multimodal/completions",
                        json={"prompt": "x", "images": ["not!!base64"]})
        assert r.status_code == 400


def test_multimodal_endpoint_requires_an_image(monkeypatch):
    srv, client = _client(monkeypatch)
    with client:
        srv._STATE["mm_encoder"] = CLIPModalityEncoder(
            d_model=srv._STATE["model"].config.d_model, tower=_StubTower()
        ).eval()
        srv._STATE["mm_ready"] = True
        r = client.post("/v1/multimodal/completions",
                        json={"prompt": "x", "images": []})
        assert r.status_code == 400


# --- genuine CLIP path (skips offline / without transformers) -------------

@pytest.mark.slow
def test_real_clip_vision_tower_smoke():
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    from mt_lnn.multimodal import CLIPVisionTower
    try:
        tower = CLIPVisionTower("openai/clip-vit-base-patch32")
    except Exception as e:  # offline / download blocked / missing weights
        pytest.skip(f"CLIP weights unavailable: {e}")
    from PIL import Image
    feats = tower.encode([Image.new("RGB", (224, 224), (0, 128, 255))])
    assert feats.dim() == 3 and feats.shape[0] == 1
    assert feats.shape[-1] == tower.hidden_size
