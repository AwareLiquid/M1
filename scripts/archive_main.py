"""archive_main.py — 等云 GPU 实例恢复后自动归档主要数据。

清单:
  1. checkpoints/ckpt_120000.pt   (20.3GB, 最佳 val PPL 77.09)
  2. checkpoints/ckpt_100000.pt   (20.3GB, 次佳 val PPL 79.64)
  3. 结果文件(小): curriculum jsonl / 日志 / moe json / p1 summary

用法: py -3.11 scripts/archive_main.py
轮询间隔 300s，实例恢复后自动执行下载并写报告。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_remote import connect, download  # noqa: E402

LOCAL_ROOT = Path("E:/M1/archived")
SMALL_FILES = [
    "/home/user/M1/curriculum_results.jsonl",
    "/home/user/M1/curriculum_results_amp.jsonl",
    "/home/user/llm2b_cont.log",
    "/home/user/llm2b_cont2.log",
    "/home/user/llm2b_cont3.log",
    "/home/user/M3/experiments/results/moe_g300m.json",
    "/home/user/M3/experiments/results/exp_p1/summary.json",
    "/home/user/M1/curriculum.log",
]
BIG_FILES = [
    ("/home/user/M1/checkpoints/ckpt_120000.pt", "ckpt_120000.pt"),
    ("/home/user/M1/checkpoints/ckpt_100000.pt", "ckpt_100000.pt"),
]

POLL_SECONDS = 300
MAX_POLLS = 288  # 24h


def main():
    print(f"[archive] polling SSH every {POLL_SECONDS}s "
          f"(max {MAX_POLLS} polls) ...", flush=True)
    for poll in range(MAX_POLLS):
        try:
            c = connect()
        except Exception as e:
            print(f"[{poll}] SSH not up yet ({type(e).__name__}), "
                  f"retry in {POLL_SECONDS}s", flush=True)
            time.sleep(POLL_SECONDS)
            continue
        print("[archive] instance is up — downloading ...", flush=True)
        # 1. 小文件（结果记录）
        for remote in SMALL_FILES:
            local = LOCAL_ROOT / remote.replace("/home/user/", "").replace(
                "/M1/", "m1/").replace("/M3/", "m3/")
            try:
                download(c, remote, str(local))
            except Exception as e:
                print(f"  skip {remote}: {e}", flush=True)
        # 2. 大 checkpoint
        for remote, name in BIG_FILES:
            try:
                download(c, remote, str(LOCAL_ROOT / name))
            except Exception as e:
                print(f"  FAILED {name}: {e}", flush=True)
        c.close()
        print("[archive] done. see E:/M1/archived/", flush=True)
        return
    print("[archive] gave up after 24h of polling", flush=True)


if __name__ == "__main__":
    main()
