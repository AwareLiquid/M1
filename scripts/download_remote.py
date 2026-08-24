"""download_remote.py — 从云 GPU 下载大文件到本地（SFTP，进度 + 分块重试）。

用法:
  py -3.11 scripts/download_remote.py /home/user/M1/checkpoints/ckpt_120000.pt E:\\M1\\checkpoints\\ckpt_120000.pt
  py -3.11 scripts/download_remote.py --list /home/user/M1/checkpoints/   (列出远端文件)
"""
import argparse
import os
import sys
import time

import paramiko

HOST = "user@65.49.232.182:10012"
PASSWORD = "DD6ydue8leCNT/nA"


def connect():
    host, port = HOST.rsplit(":", 1)
    user = host.split("@")[0]
    host = host.split("@")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=int(port), username=user, password=PASSWORD,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def list_remote(c, path):
    sftp = c.open_sftp()
    try:
        attrs = sftp.listdir_attr(path)
    except FileNotFoundError:
        print(f"not found: {path}")
        return
    for a in attrs:
        print(f"{a.st_size:>14,}  {path.rstrip('/')}/{a.filename}")
    sftp.close()


def download(c, remote, local, chunk=8 * 1024 * 1024, retries=5):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    sftp = c.open_sftp()
    size = sftp.stat(remote).st_size
    start = os.path.getsize(local) if os.path.exists(local) else 0
    mode = "ab" if start else "wb"
    if start >= size:
        print(f"already complete: {local} ({size:,} bytes)")
        sftp.close()
        return
    print(f"downloading {remote} ({size:,} bytes) -> {local}"
          f" [resume at {start:,}]")
    t0 = time.time()
    with open(local, mode) as f:
        while start < size:
            ok = False
            for attempt in range(retries):
                try:
                    with sftp.open(remote, "rb", bufsize=chunk) as rf:
                        rf.seek(start)
                        data = rf.read(min(chunk, size - start))
                    f.write(data)
                    start += len(data)
                    ok = True
                    break
                except (EOFError, OSError, paramiko.SSHException) as e:
                    print(f"  retry {attempt+1}/{retries} after {e}",
                          flush=True)
                    time.sleep(5)
                    sftp.close()
                    sftp = c.open_sftp()
            if not ok:
                print("FAILED after retries")
                sftp.close()
                return
            if start % (chunk * 16) < chunk:
                pct = start / size * 100
                spd = start / max(time.time() - t0, 1e-3) / 1e6
                print(f"  {pct:.1f}%  {start:,}/{size:,}  {spd:.1f} MB/s",
                      flush=True)
    dt = time.time() - t0
    print(f"done: {local} in {dt/60:.1f} min"
          f" ({size/dt/1e6:.1f} MB/s)")
    sftp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("remote")
    ap.add_argument("local", nargs="?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    c = connect()
    if args.list:
        list_remote(c, args.remote)
        return
    if not args.local:
        print("need LOCAL path")
        return
    download(c, args.remote, args.local)
    c.close()


if __name__ == "__main__":
    main()
