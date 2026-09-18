#!/usr/bin/env python3
"""tail_kaggle_log — 实时调取 Kaggle kernel 日志(含 RUNNING 会话)。

绕开 CLI `kernels logs`(只读已完成会话)的限制, 直用官方流式端点
GET /api/v1/kernels/logs/stream/{owner}/{slug}: 会话运行中时它是上游 SSE
直播, 结束后自动回退为落盘日志。Basic auth 走 ~/.kaggle/access_token。

用法:
  .venv/bin/python tools/tail_kaggle_log.py            # 探雷, 跟到 END_OF_LOG
  ... --kernel OWNER/SLUG --from-time 12000            # 指定 kernel/跳过前 N 秒
  ... --once                                           # 打最新 15 条 stdout 就退
"""
import argparse
import json
import os
import time

import requests


def creds():
    """返回 requests auth：优先新版 KGAT 单行 token(Bearer)，兼容旧 username/key。"""
    p = os.path.expanduser("~/.kaggle/access_token")
    if os.path.exists(p):
        tok = open(p, encoding="utf-8").read().strip()
        if tok.startswith("KGAT_"):
            return ("bearer", tok)
        try:
            j = json.loads(tok)
            return (j["username"], j["key"])
        except ValueError:
            pass
    cfg = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(cfg):
        j = json.load(open(cfg, encoding="utf-8"))
        return (j["username"], j["key"])
    raise SystemExit("no kaggle credentials found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="aricredemption/m1-pc-band-scan-b-2-pointer-chase-mine")
    ap.add_argument("--from-time", type=float, default=0.0, help="只打印 time>=该值(秒) 的行")
    ap.add_argument("--once", action="store_true", help="打最后 15 条 stdout 即退")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    user, key = creds()
    url = f"https://www.kaggle.com/api/v1/kernels/logs/stream/{args.kernel}"
    headers = {"Accept": "text/event-stream, */*"}
    auth = None
    if user == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    else:
        auth = (user, key)
    t_end = time.time() + args.timeout
    buf = []
    # timeout 是总预算; 单次 read 超时(步间安静期)不算错, 继续下一轮。
    while time.time() < t_end:
        try:
            with requests.get(url, auth=auth, stream=True,
                              timeout=(15, max(5, t_end - time.time())),
                              headers=headers) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if time.time() > t_end:
                        break
                    if not raw or not raw.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(raw[5:].strip())
                    except ValueError:
                        continue
                    if ev.get("stream_name") == "stderr" and "SyntaxWarning" in ev.get("data", ""):
                        continue
                    if ev.get("data", "").strip() == "END_OF_LOG":
                        args.end = True
                        break
                    if ev.get("time", 0) >= args.from_time:
                        line = f"[{(ev['time'])/60:6.1f}min] {ev['data'].rstrip()}"
                        if args.once:
                            buf.append(line)
                        else:
                            print(line, flush=True)
            if getattr(args, "end", False):
                break
        except (requests.ConnectionError, requests.Timeout):
            if args.once:
                break  # 一次性模式: 拿到当前缓冲即可
            time.sleep(3)
    for line in buf[-15:]:
        print(line)


if __name__ == "__main__":
    main()
