"""
app.py — Gradio web demo for MT-LNN (Hugging Face Spaces).

Loads a base causal-LM from the Hub (default: Qwen2.5-0.5B-Instruct, supports
Chinese + English) and optionally applies a saved MT-LNN adapter checkpoint.
On free-CPU Spaces the model runs in fp32; on GPU it switches to bfloat16.

Environment variables (set in Space Settings → Variables):
  BASE_MODEL   HF model-id to load  (default: Qwen/Qwen2.5-0.5B-Instruct)
  ADAPTER_PATH local path or HF path to an MT-LNN adapter .pt  (optional)
"""

import os

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional self-thinking serve path. Lives in mt_lnn.thinking and imports
# only torch + mt_lnn.deliberation, so the demo degrades gracefully (the
# "Self-Thinking" tab simply hides) if the package isn't on the path — e.g.
# when app.py is deployed standalone to a Space.
try:
    from mt_lnn.thinking import (
        generate_with_thinking,
        render_trace_markdown,
        render_trace_html,
    )
    from mt_lnn.deliberation import RouterThresholds
    _THINKING_AVAILABLE = True
except Exception as _exc:  # pragma: no cover — optional feature
    print(f"[MT-LNN] self-thinking tab disabled — {_exc}")
    _THINKING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL   = os.environ.get("BASE_MODEL",   "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = (torch.bfloat16
                if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float32)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
print(f"[MT-LNN] Loading {BASE_MODEL} on {DEVICE} ({DTYPE}) …")
_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token

_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
)

if ADAPTER_PATH and os.path.isfile(ADAPTER_PATH):
    try:
        from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state
        checkpoint = torch.load(ADAPTER_PATH, map_location="cpu")
        attach_adapters_from_checkpoint(_model, checkpoint)
        load_adapter_state(_model, ADAPTER_PATH, strict=False)
        print(f"[MT-LNN] Adapter loaded from {ADAPTER_PATH}")
    except Exception as exc:
        print(f"[MT-LNN] WARNING: could not load adapter — {exc}")

if DEVICE == "cpu":
    _model = _model.to(DEVICE)
_model.eval()
print("[MT-LNN] Model ready.")

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < v[:, [-1]], float("-inf"))


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    # Shift by one so the token that CROSSES the p boundary is kept (and the
    # most probable token is always kept) — standard nucleus semantics, same
    # as serve/server_hf.py's _sample_next_token.
    remove = probs.cumsum(dim=-1) - probs > p
    keep = ~remove
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep)
    return logits.masked_fill(~mask, float("-inf"))


def _build_prompt(history: list, message: str) -> str:
    """Build a chat prompt using apply_chat_template when available."""
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    if getattr(_tokenizer, "chat_template", None):
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for models without a chat template
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{bot_msg}\n"
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"
    return prompt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    eos_id = _tokenizer.eos_token_id
    generated_ids = ids.clone()

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            out = _model(input_ids=generated_ids)
            logits = out.logits[:, -1, :] / max(float(temperature), 1e-6)
            logits = _top_k(logits, int(top_k))
            logits = _top_p(logits, float(top_p))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break

    # Decode only the newly generated tokens to avoid space/encoding issues
    new_tokens = generated_ids[0, prompt_len:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat_stream(
    message: str,
    history: list,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    prompt = _build_prompt(history, message)
    ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    eos_id = _tokenizer.eos_token_id
    generated_ids = ids.clone()

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            out = _model(input_ids=generated_ids)
            logits = out.logits[:, -1, :] / max(float(temperature), 1e-6)
            logits = _top_k(logits, int(top_k))
            logits = _top_p(logits, float(top_p))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)

            # Decode ALL new tokens together — fixes SentencePiece space-prefix loss
            new_tokens = generated_ids[0, prompt_len:]
            yield _tokenizer.decode(new_tokens, skip_special_tokens=True)

            if eos_id is not None and next_id.item() == eos_id:
                break


# ---------------------------------------------------------------------------
# Self-thinking generation (Layer 2 router → live thinking trace)
# ---------------------------------------------------------------------------

def think_generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    low_thr: float,
    high_thr: float,
    n_critique: int,
):
    """Run the deliberation-router generation and return (text, summary, html).

    The router classifies each decode step as LOCAL / SELF_CRITIQUE / CLOUD
    by next-token entropy; uncertain steps are re-decoded via a token-level
    self-consistency vote (genuine "self-thinking"). The public demo has no
    cloud oracle, so CLOUD steps are flagged and fall back to self-critique.
    """
    if not _THINKING_AVAILABLE:
        return "self-thinking unavailable (mt_lnn not importable)", "", ""
    if not prompt.strip():
        return "", "", ""
    thresholds = RouterThresholds(low=float(low_thr), high=float(high_thr))
    text, trace = generate_with_thinking(
        _model,
        _tokenizer,
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        n_critique_samples=int(n_critique),
        thresholds=thresholds,
        device=DEVICE,
    )
    return text, render_trace_markdown(trace), render_trace_html(trace)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_adapter_badge = (
    f"🧠 **MT-LNN adapter active** (`{os.path.basename(ADAPTER_PATH)}`)"
    if ADAPTER_PATH and os.path.isfile(ADAPTER_PATH)
    else "⚙️ Running vanilla base model (no MT-LNN adapter)"
)
_description = f"""
## MT-LNN — Microtubule Linear Neural Network
**Base model:** `{BASE_MODEL}` &nbsp;|&nbsp; **Device:** `{DEVICE}`

{_adapter_badge}

This demo showcases the [MT-LNN architecture](https://huggingface.co/EverestAn/MT-LNN):
a biologically-inspired hybrid that couples a standard transformer with a linear
recurrent network modelling microtubule quantum-coherence dynamics.

支持中英文对话 · Bilingual (Chinese & English) · Type below and hit **Submit**.
"""

with gr.Blocks(title="MT-LNN Demo") as demo:
    gr.Markdown(_description)

    with gr.Tab("💬 Chat"):
        gr.ChatInterface(
            fn=chat_stream,
            additional_inputs=[
                gr.Slider(32, 512, value=200, step=32,   label="Max new tokens"),
                gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature"),
                gr.Slider(0,   100, value=0,   step=1,   label="Top-k  (0 = off)"),
                gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p"),
            ],
        )

    with gr.Tab("📝 Completion"):
        prompt_box = gr.Textbox(
            lines=5, placeholder="Enter a prompt… / 输入提示词…", label="Prompt"
        )
        with gr.Row():
            max_tok  = gr.Slider(32,  512,  value=200,  step=32,   label="Max new tokens")
            temp     = gr.Slider(0.1, 2.0,  value=0.7,  step=0.05, label="Temperature")
            top_k_sl = gr.Slider(0,   100,  value=0,    step=1,    label="Top-k  (0 = off)")
            top_p_sl = gr.Slider(0.0, 1.0,  value=0.9,  step=0.05, label="Top-p")
        run_btn = gr.Button("Generate", variant="primary")
        output_box = gr.Textbox(lines=10, label="Generated text", interactive=False)
        run_btn.click(
            fn=generate_text,
            inputs=[prompt_box, max_tok, temp, top_k_sl, top_p_sl],
            outputs=output_box,
        )

    if _THINKING_AVAILABLE:
        with gr.Tab("🧠 Self-Thinking"):
            gr.Markdown(
                "**自我思考 / Self-thinking.** Each token is routed by a "
                "3-way deliberation policy: confident tokens decode locally "
                "(green); uncertain tokens are *reconsidered* via a "
                "self-consistency vote (orange, underlined if revised); "
                "tokens needing an external fact are flagged for the cloud "
                "(red). Hover any token to see its entropy.\n\n"
                "_Policy lives in `mt_lnn/deliberation.py`; the live decode "
                "mechanism + trace in `mt_lnn/thinking.py` — zero coupling to "
                "the backbone._"
            )
            think_prompt = gr.Textbox(
                lines=4, label="Prompt",
                placeholder="Ask something the model may be unsure about…",
            )
            with gr.Row():
                think_max  = gr.Slider(16, 256, value=96, step=16, label="Max new tokens")
                think_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
                think_topp = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
            with gr.Row():
                think_low  = gr.Slider(0.5, 5.0, value=3.0, step=0.1,
                                       label="Low entropy threshold (→ local)")
                think_high = gr.Slider(1.0, 8.0, value=5.0, step=0.1,
                                       label="High entropy threshold (→ cloud)")
                think_nc   = gr.Slider(2, 8, value=3, step=1,
                                       label="Self-critique samples")
            think_btn = gr.Button("Think & Generate", variant="primary")
            think_out = gr.Textbox(lines=6, label="Response", interactive=False)
            think_summary = gr.Markdown()
            think_html = gr.HTML(label="Thinking trace")
            think_btn.click(
                fn=think_generate,
                inputs=[think_prompt, think_max, think_temp, think_topp,
                        think_low, think_high, think_nc],
                outputs=[think_out, think_summary, think_html],
            )

    gr.Markdown(
        "---\n"
        "Model weights & code: [EverestAn/MT-LNN](https://huggingface.co/EverestAn/MT-LNN) · "
        "MIT license"
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(),
    ssr_mode=False,
)
