#!/usr/bin/env python3
"""Kaggle launcher for the M2-P0 C1 global-head sweep.

The Kaggle T4 x2 runtime is used as two independent workers. Each worker gets
one visible GPU and a disjoint subset of global-head quotas. Results are kept
per quota before being merged, so parallelism does not change experiment
semantics or duplicate JSONL rows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


OUT = Path("/kaggle/working/c1_results")
REPO_URL = "https://github.com/everest-an/M1.git"
DEFAULT_STEPS = 30_000
SEEDS = (0, 1, 2)
EVAL_DEPTHS = (1, 2, 4)
WORKER_SPECS = (
    (0, 0, (0, 1)),
    (1, 1, (2, 4)),
)


def run(
    cmd: list[str],
    *,
    label: str,
    cwd: Path | None = None,
    log: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    started = time.monotonic()
    print(f"[{label}] START command={' '.join(cmd)}", flush=True)
    sink = log.open("w", encoding="utf-8") if log is not None else None
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if sink is not None:
                sink.write(line)
                sink.flush()
            print(f"[{label}] {line}", end="", flush=True)
        returncode = process.wait()
    finally:
        if sink is not None:
            sink.close()
    elapsed = time.monotonic() - started
    print(f"[{label}] EXIT returncode={returncode} elapsed_s={elapsed:.1f}", flush=True)
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_worker(worker_id: int, gpu_id: int, heads: list[int]) -> list[Path]:
    worker_started = time.monotonic()
    repo = Path(f"/tmp/M1_c1_worker_{worker_id}")
    print(
        f"WORKER_START worker={worker_id} gpu={gpu_id} heads={heads} time={time.time():.0f}",
        flush=True,
    )
    if repo.exists():
        print(f"WORKER_CLEANUP worker={worker_id} path={repo}", flush=True)
        shutil.rmtree(repo)
    clone_log = OUT / f"worker_{worker_id}_clone.log"
    run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo)],
        label=f"worker={worker_id} phase=clone",
        log=clone_log,
    )
    print(f"WORKER_CLONED worker={worker_id} path={repo}", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    result_files: list[Path] = []
    for head in heads:
        log = OUT / f"global_heads_{head}.log"
        result = repo / "benchmarks/results/reasoning_depth.jsonl"
        previous_size = result.stat().st_size if result.exists() else 0
        cmd = [
            sys.executable,
            "benchmarks/reasoning_depth.py",
            "--task", "pointer_chase",
            "--difficulty", "4",
            "--n_values", "8",
            "--steps", os.environ.get("C1_STEPS", str(DEFAULT_STEPS)),
            "--seeds", *(str(seed) for seed in SEEDS),
            "--mode", "fixed",
            "--eval_depths", *(str(depth) for depth in EVAL_DEPTHS),
            "--stack",
            "--mix",
            "--n_global_heads", str(head),
            "--tag", f"kaggle-c1-global-heads-{head}",
        ]
        print(
            f"HEAD_START worker={worker_id} gpu={gpu_id} head={head} "
            f"steps={cmd[cmd.index('--steps') + 1]} seeds={SEEDS} eval_depths={EVAL_DEPTHS} "
            f"time={time.time():.0f}",
            flush=True,
        )
        run(cmd, cwd=repo, log=log, env=env, label=f"worker={worker_id} head={head}")
        if not result.exists() or result.stat().st_size <= previous_size:
            raise RuntimeError(f"no new result rows for head={head}: {result}")
        per_head = OUT / f"global_heads_{head}.jsonl"
        with result.open("rb") as source, per_head.open("wb") as target:
            source.seek(previous_size)
            shutil.copyfileobj(source, target)
        result_files.append(per_head)
        print(
            f"HEAD_DONE worker={worker_id} head={head} elapsed_s={time.monotonic() - worker_started:.1f} "
            f"new_result_bytes={per_head.stat().st_size}",
            flush=True,
        )
    print(
        f"WORKER_DONE worker={worker_id} gpu={gpu_id} elapsed_s={time.monotonic() - worker_started:.1f}",
        flush=True,
    )
    return result_files


def main() -> None:
    import torch

    run_started = time.monotonic()
    expected_gpus = len(WORKER_SPECS)
    if not torch.cuda.is_available() or torch.cuda.device_count() < expected_gpus:
        raise RuntimeError(f"C1 requires at least {expected_gpus} visible GPUs")
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda={torch.version.cuda} gpus={torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"RUN_CONFIG workers={WORKER_SPECS} steps={os.environ.get('C1_STEPS', DEFAULT_STEPS)} "
        f"seeds={SEEDS} eval_depths={EVAL_DEPTHS} output={OUT}",
        flush=True,
    )
    print(f"RUN_START time={time.time():.0f}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_worker, *spec) for spec in WORKER_SPECS]
        result_files = [path for future in futures for path in future.result()]

    with (OUT / "reasoning_depth.jsonl").open("w", encoding="utf-8") as combined:
        for result in result_files:
            if not result.exists():
                raise RuntimeError(f"missing expected result file: {result}")
            combined.write(result.read_text(encoding="utf-8"))
    print(
        f"RUN_DONE elapsed_s={time.monotonic() - run_started:.1f} "
        f"result_files={[str(path) for path in result_files]} combined={OUT / 'reasoning_depth.jsonl'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
