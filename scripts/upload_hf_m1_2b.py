"""upload_hf_m1_2b.py — M1-2B checkpoint → Hugging Face (AwareLiquid/M1-2B)

前置:
  pip install huggingface_hub
  认证: HF_TOKEN 环境变量, 或先 `huggingface-cli login`

用法:
  py -3.11 scripts/upload_hf_m1_2b.py E:\\M1\\checkpoints\\ckpt_120000.pt --dry-run
  py -3.11 scripts/upload_hf_m1_2b.py E:\\M1\\checkpoints\\ckpt_120000.pt

--dry-run 只校验本地文件 + 打印计划, 不触碰远端。README model card 由
RELEASE_README 模板渲染后一并上传 (仓库即开箱可见文档)。
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_REPO = "AwareLiquid/M1-2B"

# 发布时替换 {REPLACE_*} 占位符。诚实边界必须保留。
RELEASE_README = """---
language:
- en
license: mit
tags:
- awareliquid
- mt-lnn
- liquid-neural-network
- hybrid-architecture
pipeline_tag: text-generation
---

# M1-2B

1.9B-parameter AwareLiquid hybrid — window attention + liquid core
(selective-decay, exp parameterization) in every layer. {REPLACE_LAYERS}
layers, d_model {REPLACE_DMODEL}. Trained from scratch.

**Status at upload:** research milestone — {REPLACE_STATUS_NOTE}

## Serve it

```bash
# drop the .pt into checkpoints/, then:
CKPT_PATH=checkpoints/{REPLACE_FILENAME} TOKENIZER=gpt2 \\
  python -m uvicorn serve.server:app --port 8000
```

> 1.9B fp32 ≈ 7.6 GB weights; CPU inference works but is slow.
> A GPU instance is recommended for interactive use.

## Measured

{REPLACE_MEASURED}

## Boundaries (honest)

{REPLACE_BOUNDARIES}

License: MIT.
"""


def build_readme(cfg: dict) -> str:
    return (RELEASE_README
            .replace("{REPLACE_LAYERS}", str(cfg["n_layers"]))
            .replace("{REPLACE_DMODEL}", str(cfg["d_model"]))
            .replace("{REPLACE_FILENAME}", cfg["filename"])
            .replace("{REPLACE_STATUS_NOTE}", cfg["status_note"])
            .replace("{REPLACE_MEASURED}", cfg["measured"])
            .replace("{REPLACE_BOUNDARIES}", cfg["boundaries"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="本地 checkpoint .pt 路径")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-layers", type=int, default=35)
    ap.add_argument("--d-model", type=int, default=2912)
    ap.add_argument("--status-note", default="checkpoint published before full "
                    "convergence so the run is reproducible; see boundaries below.")
    ap.add_argument("--measured", default=(
        "- O(1) carried state (constant-size recurrent state, no growing KV-cache)\n"
        "- Training curve: see the GitHub Release notes for the exact PPL trajectory\n"))
    ap.add_argument("--boundaries", default=(
        "- This is a mid-training checkpoint, not a converged production model.\n"
        "- Language-modeling quality is not yet at the level of established "
        "open LLMs; text output will be low-quality.\n"
        "- Published for research reproducibility and roadmap transparency, "
        "not as a recommended inference model.\n"))
    args = ap.parse_args()

    path = Path(args.ckpt)
    assert path.exists(), f"not found: {path}"
    size_gb = path.stat().st_size / 1e9

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"HF 身份: {who['name']}")
    print(f"计划: {args.repo} (public, MIT)")
    print(f"文件: {path} ({size_gb:.2f} GB)")
    cfg = {
        "n_layers": args.n_layers, "d_model": args.d_model,
        "filename": path.name, "status_note": args.status_note,
        "measured": args.measured, "boundaries": args.boundaries,
    }
    readme = build_readme(cfg)
    print("README 预览:")
    print("───")
    print(readme)
    print("───")

    if args.dry_run:
        print("\n[dry-run] 未上传任何内容。")
        return 0

    print(f"\n创建仓库 {args.repo} ...")
    api.create_repo(args.repo, private=False, license="mit", exist_ok=True)
    print(f"上传 {path.name} (断点续传) ...")
    api.upload_file(
        path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.repo,
        commit_message=f"upload {path.name} ({datetime.now(timezone.utc):%Y-%m-%d})",
    )
    print("上传 README.md ...")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
        repo_id=args.repo, commit_message="add model card",
    )
    print(f"完成: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
