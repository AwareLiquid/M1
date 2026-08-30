"""三行代码路径 vs 自研 serve 路径：冷加载 / 依赖足迹 / 首 token 延迟。

对照的两条路径
--------------
A. **HF 原生（本分支新增）**：``MTLNNForCausalLM.from_pretrained(dir)``
   —— 产物是 config.json + safetensors，可被 Auto 类、Hub、vLLM/SGLang 的
   Auto 分发识别。
B. **自研 serve（既有）**：``serve/server.py::_build_model`` 的口径
   —— ``MTLNNModel(MTLNNConfig(**ckpt["config"]))`` + ``torch.load`` 的
   ``model_state``。

口径局限（写死在 JSON 的 ``caveats`` 里，不要只写在文档里）
----------------------------------------------------------
1. **CPU、随机小权重**（vocab 64 / 2 层 / d_model 104）。绝对数字不可外推到
   真实模型：真实瓶颈是权重带宽与 GPU kernel，这里两者都不存在。
2. **冷加载含解释器与 import**。用独立子进程测，所以数字里包含
   ``python`` 启动 + ``import torch``（通常 1~3 s，占大头）。这正是我们要暴露
   的事实：HF 路径的额外成本主要来自 transformers 生态的 import，而不是
   序列化格式本身。
3. **首 token 延迟**是 batch=1、prompt=8 token、greedy、单次前向的口径，
   不含 CUDA graph / torch.compile / 连续批处理 / PagedAttention。
4. **依赖足迹用 ``sys.modules`` 计数近似**，不等于 pip 安装体积，也不等于
   磁盘占用；它衡量的是"import 之后进程里多了多少个模块"。
5. 两条路径加载的是**同一份随机权重**，所以差异只来自加载路径，不来自数学。

用法::

    python benchmarks/hf_load_bench.py [--repeats 5] [--out benchmarks/results/hf_load_bench.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.hf_model import MTLNNForCausalLM, tiny_o_config, tiny_o_model
from mt_lnn.model import MTLNNModel

DEFAULT_OUT = "benchmarks/results/hf_load_bench.json"
PROMPT_TOKENS = 8
# 子进程要走 sys.path，不能依赖 cwd —— 从别处调用脚本时 cwd 未必是仓库根。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOTSTRAP = f"import sys; sys.path.insert(0, {ROOT!r}); "

CAVEATS = [
    "CPU only, random small weights (vocab=64, n_layers=2, d_model=104) — "
    "absolute numbers do NOT extrapolate to real checkpoints.",
    "Cold-load timings are measured in a FRESH subprocess, so they include "
    "python startup + `import torch` (usually the dominant term).",
    "First-token latency is batch=1, prompt=8 tokens, greedy, single forward "
    "— no CUDA graph, no torch.compile, no continuous batching.",
    "Dependency footprint counts entries in sys.modules; it approximates "
    "runtime import surface, not pip download size or disk usage.",
    "Both paths load the SAME random weights, so any difference comes from the "
    "loading path, not from the math.",
]


def main(argv=None) -> int:
    """跑三条测量、落盘 JSON、打印摘要。"""
    args = _parse_args(argv)
    workdir = tempfile.mkdtemp(prefix="hf_load_bench_")
    hf_dir = os.path.join(workdir, "hf_native")
    ckpt_path = os.path.join(workdir, "serve_ckpt.pt")
    source = tiny_o_model(seed=args.seed)
    source.save_pretrained(hf_dir)
    _save_serve_checkpoint(source, ckpt_path)
    report = {
        "meta": _meta(args),
        "caveats": CAVEATS,
        "cold_load_s": _bench_cold_load(hf_dir, ckpt_path, args.repeats),
        "dependency_footprint": _bench_dependency_footprint(),
        "first_token_ms": _bench_first_token(hf_dir, ckpt_path, args),
    }
    _write_report(report, args.out)
    _print_summary(report)
    return 0


def _bench_cold_load(hf_dir: str, ckpt_path: str, repeats: int) -> dict:
    """子进程冷加载：HF 原生 vs 自研 serve，各 repeats 次取中位数。"""
    hf_code = (
        "import time;" + BOOTSTRAP +
        "t=time.perf_counter(); "
        "from mt_lnn.hf_model import MTLNNForCausalLM; "
        f"m=MTLNNForCausalLM.from_pretrained({hf_dir!r}); "
        "print(time.perf_counter()-t)"
    )
    serve_code = (
        "import time,dataclasses,torch;" + BOOTSTRAP +
        "from mt_lnn.config import MTLNNConfig; from mt_lnn.model import MTLNNModel; "
        "t=time.perf_counter(); "
        f"ck=torch.load({ckpt_path!r}, map_location='cpu', weights_only=False); "
        "valid={f.name for f in dataclasses.fields(MTLNNConfig) if f.init}; "
        "m=MTLNNModel(MTLNNConfig(**{k:v for k,v in ck['config'].items() if k in valid})); "
        "m.load_state_dict(ck['model_state'], strict=False); "
        "print(time.perf_counter()-t)"
    )
    return {
        "repeats": repeats,
        "hf_native": _timed_subprocess(hf_code, repeats),
        "custom_serve": _timed_subprocess(serve_code, repeats),
    }


def _bench_dependency_footprint() -> dict:
    """import 面积：裸 torch / +mt_lnn / +transformers 三层，逐层做差。"""
    base = _module_count("import torch")
    native = _module_count(BOOTSTRAP + "import mt_lnn.model, mt_lnn.config")
    hf = _module_count(BOOTSTRAP + "import mt_lnn.hf_model")
    return {
        "modules_torch": base,
        "modules_plus_mtlnn": native,
        "modules_plus_transformers": hf,
        "delta_mtlnn": native - base,
        "delta_transformers": hf - native,
        "versions": _versions(),
    }


def _bench_first_token(hf_dir: str, ckpt_path: str, args) -> dict:
    """首 token 延迟：两条路径各生成 1 个 token，warmup 后取中位数。"""
    ids = torch.randint(0, int(tiny_o_config().vocab_size), (1, PROMPT_TOKENS))
    hf_model = MTLNNForCausalLM.from_pretrained(hf_dir).eval()
    serve_model = _load_serve_checkpoint(ckpt_path).eval()
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "repeats": args.repeats,
        "hf_native_ms": _median_first_token(hf_model, ids, args.repeats),
        "custom_serve_ms": _median_first_token(serve_model, ids, args.repeats),
        "logits_max_abs_diff": _logits_gap(hf_model, serve_model, ids),
    }


def _median_first_token(model, ids: torch.Tensor, repeats: int) -> float:
    """warmup 一次后重复计时；greedy 生成 1 个 token（= 一次前向 + 采样）。"""
    with torch.no_grad():
        model.generate(ids, max_new_tokens=1, do_sample=False)
        samples = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            model.generate(ids, max_new_tokens=1, do_sample=False)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _logits_gap(model_a, model_b, ids: torch.Tensor) -> float:
    """两条路径的 logits 最大绝对差——数学零改动的直接证据（应为 0.0）。"""
    with torch.no_grad():
        a = model_a(ids).logits
        b = model_b(input_ids=ids)["logits"]
    return float((a - b).abs().max().item())


def _timed_subprocess(code: str, repeats: int) -> dict:
    """跑 ``python -c code`` repeats 次，同时记墙钟与"进程内"两个口径。

    子进程打印的是它自己测到的加载耗时（不含解释器启动），墙钟则包含
    全部。两者之差就是"启动 + import"的固定成本——HF 路径比自研 serve
    路径多付的主要就是这一项。
    """
    wall, inner = [], []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        wall.append(time.perf_counter() - t0)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip()[-800:]}
        inner.append(float(proc.stdout.strip().splitlines()[-1]))
    return {
        "wall_median_s": statistics.median(wall),
        "wall_min_s": min(wall),
        "load_only_median_s": statistics.median(inner),
        "startup_plus_import_s": statistics.median(wall) - statistics.median(inner),
    }


def _module_count(code: str) -> int:
    """子进程 import 之后 ``sys.modules`` 的条目数。"""
    proc = subprocess.run(
        [sys.executable, "-c", f"{code}; print(len(__import__('sys').modules))"],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        return -1
    return int(proc.stdout.strip().splitlines()[-1])


def _versions() -> dict:
    out = {"torch": torch.__version__}
    for name in ("transformers", "peft", "safetensors"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def _save_serve_checkpoint(wrapper: MTLNNForCausalLM, path: str) -> None:
    """按 serve/server.py::_build_model 期望的格式落盘。"""
    native = wrapper.model
    valid = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
    payload = {
        "config": {k: getattr(native.config, k) for k in valid},
        "model_state": native.state_dict(),
        "step": 0,
    }
    torch.save(payload, path)


def _load_serve_checkpoint(path: str) -> MTLNNModel:
    """serve/server.py::_build_model 的 checkpoint 分支（不含 FastAPI 部分）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    valid = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
    model = MTLNNModel(MTLNNConfig(**{k: v for k, v in ckpt["config"].items() if k in valid}))
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def _meta(args) -> dict:
    return {
        "generated_by": "benchmarks/hf_load_bench.py",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "seed": args.seed,
        "repeats": args.repeats,
        "weights": "random-init tiny_o_config (see mt_lnn.hf_model)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_report(report: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[hf_load_bench] wrote {out_path}")


def _print_summary(report: dict) -> None:
    cold = report["cold_load_s"]
    dep = report["dependency_footprint"]
    ftt = report["first_token_ms"]
    hf_cold, serve_cold = cold["hf_native"], cold["custom_serve"]
    print(f"[hf_load_bench] cold load  hf.wall={hf_cold['wall_median_s']:.3f}s "
          f"(load_only={hf_cold['load_only_median_s']:.3f}s)  "
          f"serve.wall={serve_cold['wall_median_s']:.3f}s "
          f"(load_only={serve_cold['load_only_median_s']:.3f}s)")
    print(f"[hf_load_bench] modules    torch={dep['modules_torch']} "
          f"+mtlnn={dep['delta_mtlnn']} +transformers={dep['delta_transformers']}")
    print(f"[hf_load_bench] first tok  hf={ftt['hf_native_ms']:.2f}ms "
          f"serve={ftt['custom_serve_ms']:.2f}ms "
          f"logits_gap={ftt['logits_max_abs_diff']:.3e}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repeats", type=int, default=5, help="每个测量重复次数")
    p.add_argument("--seed", type=int, default=0, help="随机小权重种子")
    p.add_argument("--out", default=DEFAULT_OUT, help="JSON 落盘路径")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
