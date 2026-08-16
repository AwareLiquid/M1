"""云 GPU 远程执行助手（paramiko）。

用法:
  py -3.11 scripts/remote_exec.py "命令"
  py -3.11 scripts/remote_exec.py "命令" --timeout 3600 --workdir /root/M1
输出: 命令的 stdout/stderr。

A100 80GB 实例: user@65.49.232.182:10012（4x24h 权限）
"""
import argparse
import sys

import paramiko

HOST = "65.49.232.182"
PORT = 10012
USER = "user"
PASSWORD = "DD6ydue8leCNT/nA"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default=None, help="远程命令")
    p.add_argument("--cmd-file", default=None,
                   help="从本地文件读取远程命令（避免 shell 引号问题）")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--workdir", default=None,
                   help="先 cd 到该目录再执行")
    p.add_argument("--upload", default=None, nargs=2, metavar=("LOCAL", "REMOTE"),
                   help="上传本地文件到远程路径（SFTP）")
    args = p.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, PORT, USER, PASSWORD, timeout=30)
    except Exception as e:
        print(f"CONNECT FAILED: {e}")
        sys.exit(1)

    if args.upload:
        local_path, remote_path = args.upload
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        print(f"uploaded {local_path} -> {remote_path}")
        client.close()
        return

    if args.cmd_file:
        with open(args.cmd_file, encoding="utf-8") as f:
            cmd = f.read().strip()
    elif args.cmd:
        cmd = args.cmd
    else:
        p.error("需要 cmd 或 --cmd-file")

    if args.workdir:
        cmd = f"cd {args.workdir} && {cmd}"

    stdin, stdout, stderr = client.exec_command(cmd, timeout=args.timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    client.close()

    if out:
        # ASCII 安全输出（Windows cp1252 控制台无法打印部分 Unicode）
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(safe)
    if err:
        safe_err = err.encode("ascii", errors="replace").decode("ascii")
        print(f"[stderr] {safe_err}", file=sys.stderr)
    print(f"[exit={code}]")


if __name__ == "__main__":
    main()
