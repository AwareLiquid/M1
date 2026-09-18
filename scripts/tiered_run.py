#!/usr/bin/env python3
"""M1 多级算力层（B-24 v1）— 参考 SkyPilot/dstack/Modal 的分层模式.

分层（对应社区谱系，见 docs/COMPUTE_TIERS.md）:
  Job      声明式作业规格（dstack 式 JSON：name/cmd/device/est_hours/...）
  Backend  适配器基类：validate() -> launch() -> status() -> harvest()
           （SkyPilot backend 抽象；Local/KaggleCPU/KaggleT4/A100SSH 四实现）
  Router   策略层（资源预估 → 层选择；L008/L010 铁律内置为检查项）

CLI:
  tiered_run.py plan    --job jobs/b2.json          # 只打印路由与命令
  tiered_run.py run     --job jobs/b2.json          # 路由 + launch
  tiered_run.py status  --backend kaggle-cpu --slug aricredemption/m1-b2-tau-ladder-cpu
  tiered_run.py harvest --backend kaggle-cpu --slug ... [--dest benchmarks/results]

零第三方依赖；T3 依赖 ~/.ssh/config 的 m1-a100 别名与 /root/M1_run 部署。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time

KAGGLE = os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "kaggle")


# ---------------------------------------------------------------------------
# Job — 声明式规格（dstack 式）
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, name: str, cmd: str, device: str = "cpu",
                 est_hours: float = 1.0, require_a100: bool = False,
                 kaggle_dir: str | None = None, resume_safe: bool = True,
                 slug: str | None = None,
                 harvest_dest: str = "benchmarks/results",
                 ckpt_dataset: str | None = None):
        self.name, self.cmd = name, cmd
        self.device, self.est_hours = device, est_hours
        self.require_a100 = require_a100
        self.kaggle_dir, self.resume_safe = kaggle_dir, resume_safe
        self.slug = slug
        self.harvest_dest = harvest_dest
        self.ckpt_dataset = ckpt_dataset

    @classmethod
    def from_json(cls, path: str) -> "Job":
        return cls(**json.load(open(path)))

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# Backend — 适配器（SkyPilot backend 抽象的最小版）
# ---------------------------------------------------------------------------

class Backend:
    name = "abstract"
    desc = "abstract"

    def describe(self) -> str:
        return self.desc

    def validate(self, job: Job) -> list[str]:
        return []

    def launch(self, job: Job, dry: bool) -> int:
        raise NotImplementedError

    def status(self, job: Job) -> str:
        raise NotImplementedError

    def harvest(self, job: Job, dest: str, dry: bool) -> int:
        raise NotImplementedError


class LocalBackend(Backend):
    name = "local"
    desc = "T0 本机：est<=1h 快速校验（L008：超 1h 禁止本机跑）"

    def validate(self, job: Job) -> list[str]:
        return (["L008: est>1h 禁止本机执行"] if job.est_hours > 1 else [])

    def launch(self, job: Job, dry: bool) -> int:
        print(f"[local] {job.cmd}")
        if dry:
            return 0
        t0 = time.time()
        rc = subprocess.run(job.cmd, shell=True).returncode
        print(f"[local] rc={rc} wall={(time.time()-t0)/60:.1f}min")
        return rc

    def status(self, job: Job) -> str:
        return "local 作业随会话结束，无独立状态"

    def harvest(self, job: Job, dest: str, dry: bool) -> int:
        print("[local] 产物就地，无需收割")
        return 0


class KaggleBackend(Backend):
    """T1(kaggle-cpu, 12h 窗) / T2(kaggle-t4, 9h 窗+周配额) 共用适配器."""

    def __init__(self, name: str, window: str, quota: str, desc: str):
        self.name = name
        self.window, self.quota = window, quota
        self.desc = desc

    def describe(self) -> str:
        return f"{self.name}: {self.window} 硬窗，{self.quota}"

    def validate(self, job: Job) -> list[str]:
        problems = []
        if not job.kaggle_dir:
            return ["缺 kaggle_dir（kaggle_kernels/<name>/）"]
        meta_path = os.path.join(job.kaggle_dir, "kernel-metadata.json")
        if not os.path.exists(meta_path):
            return [f"缺 {meta_path}"]
        meta = open(meta_path).read()
        if self.name == "kaggle-cpu" and '"enable_gpu": true' in meta:
            problems.append("T1 不应启用 GPU（配额），应 enable_gpu=false")
        # L010: launcher 必须显式钉死线程（env 钉死跨进程继承；set_num_threads 不跨进程）
        pinned = any(
            "OMP_NUM_THREADS" in open(os.path.join(job.kaggle_dir, f)).read()
            for f in os.listdir(job.kaggle_dir) if f.endswith(".py"))
        if not pinned:
            problems.append("L010: launcher 未钉死 OMP_NUM_THREADS（子进程超订陷阱）")
        return problems

    def launch(self, job: Job, dry: bool) -> int:
        problems = self.validate(job)
        if problems:
            for p in problems:
                print(f"[{self.name}] ✗ {p}", file=sys.stderr)
            return 2
        push_dir = job.kaggle_dir
        if job.ckpt_dataset:
            st = subprocess.run([KAGGLE, "datasets", "status", job.ckpt_dataset],
                                capture_output=True, text=True)
            out = st.stdout + st.stderr
            if st.returncode != 0 or "403" in out or "not found" in out.lower():
                # checkpoint 数据集尚未建成（如 DNS/网络）——剥离挂载引用，
                # 否则 push 会被 Kaggle 拒绝；代价是本会话无跨会话 resume。
                import shutil, tempfile
                tmp = tempfile.mkdtemp(prefix="kgl_nockpt_")
                for f in os.listdir(job.kaggle_dir):
                    fp = os.path.join(job.kaggle_dir, f)
                    if os.path.isfile(fp):
                        shutil.copy(fp, tmp)
                mp = os.path.join(tmp, "kernel-metadata.json")
                meta = json.load(open(mp))
                meta["dataset_sources"] = [s for s in meta.get("dataset_sources", [])
                                           if "tau-ladder-ckpt" not in s]
                json.dump(meta, open(mp, "w"), indent=2, ensure_ascii=False)
                print(f"[{self.name}] ⚠ ckpt 数据集不可用，本会话剥离挂载（resume 降级）")
                push_dir = tmp
        print(f"[{self.name}] push {push_dir}（窗 {self.window}；"
              f"resume_safe={job.resume_safe}）")
        if dry:
            return 0
        rc = subprocess.run([KAGGLE, "kernels", "push", "-p", push_dir]).returncode
        print(f"[{self.name}] 收割命令: tiered_run.py harvest --backend {self.name} "
              f"--slug {self._slug(job)} --dest benchmarks/results")
        return rc

    def _slug(self, job: Job) -> str:
        if job.slug:
            return job.slug
        meta = json.load(open(os.path.join(job.kaggle_dir, "kernel-metadata.json")))
        return meta["id"]

    def status(self, job: Job) -> str:
        r = subprocess.run([KAGGLE, "kernels", "status", self._slug(job)],
                           capture_output=True, text=True)
        return (r.stdout + r.stderr).strip().splitlines()[-1]

    def harvest(self, job: Job, dest: str, dry: bool) -> int:
        slug = self._slug(job)
        pull = [KAGGLE, "kernels", "output", slug, "-p", "/tmp/m1_harvest"]
        print(f"[{self.name}] pull {slug} -> /tmp/m1_harvest")
        if dry:
            return 0
        rc = subprocess.run(pull).returncode
        if rc:
            return rc
        import shutil
        copied = 0
        pull_dir = None
        for root, _, files in os.walk("/tmp/m1_harvest"):
            for f in files:
                if f.endswith(".json"):
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(os.path.join(root, f), os.path.join(dest, f))
                    copied += 1
                if f.endswith(".log"):
                    pull_dir = root
        print(f"[{self.name}] {copied} 个 JSON -> {dest}")
        # 多主机状态汇聚：结果 JSON 推成 checkpoint 数据集新版本，
        # 下次同 slug 启动时由 metadata.dataset_sources 挂载做 resume。
        if job.ckpt_dataset and pull_dir:
            rc_ckpt = push_checkpoint_dataset(pull_dir, job.ckpt_dataset, dry)
            if rc_ckpt:
                print(f"[{self.name}] ⚠ checkpoint 数据集推送失败（网络/DNS）——"
                      f"状态已安全在 {dest} 与 git，跨会话 resume 本轮不可用")
                return 0
            return 0
        return 0


class A100SSHBackend(Backend):
    name = "a100"
    desc = "T3 A100 40GB SSH：大 VRAM/长跑正式实验（/root/M1_run 部署约定）"

    def validate(self, job: Job) -> list[str]:
        return []

    def launch(self, job: Job, dry: bool) -> int:
        b64 = base64.b64encode(job.cmd.encode()).decode()
        remote = (f"echo {b64} | base64 -d > /root/M1_run/.job.sh && "
                  f"cd /root/M1_run && nohup bash .job.sh "
                  f"> /root/M1_run/run_$(date +%m%d_%H%M).log 2>&1 &")
        print(f"[a100] ssh {job.ssh_host} (job 经 base64，引号安全)")
        print(f"[a100] 收割: rsync -av {job.ssh_host}:/root/M1_run/benchmarks/results/ benchmarks/results/")
        if dry:
            return 0
        return subprocess.run(["ssh", job.ssh_host, remote]).returncode

    def status(self, job: Job) -> str:
        r = subprocess.run(
            ["ssh", job.ssh_host, "tail -3 /root/M1_run/$(ls -t /root/M1_run/ | grep run_ | head -1)"],
            capture_output=True, text=True)
        return (r.stdout + r.stderr).strip() or "(无日志)"

    def harvest(self, job: Job, dest: str, dry: bool) -> int:
        cmd = (f"rsync -av {job.ssh_host}:/root/M1_run/benchmarks/results/ {dest}/")
        print(f"[a100] {cmd}")
        if dry:
            return 0
        return subprocess.run(cmd, shell=True).returncode


def push_checkpoint_dataset(results_dir: str, dataset_id: str, dry: bool) -> int:
    """把 results 目录（tau_probe_*.json 等）推成/更新 Kaggle Dataset 版本.

    多主机训练状态汇聚点：harvest（本地，有凭证）写入；任意主机的新会话
    经 metadata.dataset_sources 挂载同一数据集实现跨会话/跨主机 resume。
    """
    import glob as _glob
    meta = {"title": dataset_id.split("/", 1)[1].replace("-", " "),
            "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]}
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    rows = _glob.glob(os.path.join(results_dir, "tau_probe_*.json"))
    print(f"[ckpt] {dataset_id}: {len(rows)} 个配置行，推成数据集版本")
    if dry:
        return 0
    st = subprocess.run([KAGGLE, "datasets", "status", dataset_id],
                        capture_output=True, text=True)
    exists = st.returncode == 0
    if exists:
        cmd = [KAGGLE, "datasets", "version", "-p", results_dir,
               "-m", f"harvest {time.strftime('%m%d-%H%M')}"]
    else:
        cmd = [KAGGLE, "datasets", "create", "-p", results_dir]
    rc = subprocess.run(cmd).returncode
    print(f"[ckpt] {'version' if exists else 'create'} rc={rc}")
    return rc


# 给 A100SSHBackend 一个 ssh_host 属性挂载
def _a100_with_host(host: str) -> A100SSHBackend:
    b = A100SSHBackend()
    b.ssh_host_default = host
    return b


TIERS = {
    "local": LocalBackend(),
    "kaggle-cpu": KaggleBackend("kaggle-cpu", "12h", "免费（不烧配额）",
                                "T1 Kaggle CPU：4 核 12h 窗免费，cpu 长跑（L010：4 lane × OMP 钉死）"),
    "kaggle-t4": KaggleBackend("kaggle-t4", "9h", "~30h/周配额",
                               "T2 Kaggle T4：9h 窗，~30h/周配额，小型 GPU"),
    "a100": A100SSHBackend(),
}


# ---------------------------------------------------------------------------
# Router — 策略层（资源预估 → 层）
# ---------------------------------------------------------------------------

def decide(job: Job, tier: str) -> Backend:
    if tier != "auto":
        return TIERS[tier]
    if job.device == "gpu":
        if job.require_a100 or job.est_hours > 9:
            return TIERS["a100"]
        return TIERS["kaggle-t4"]
    return TIERS["local"] if job.est_hours <= 1 else TIERS["kaggle-cpu"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="M1 多级算力层 v1 (B-24)")
    sub = p.add_subparsers(dest="action", required=True)

    def add_common(sp):
        sp.add_argument("--job", help="作业规格 JSON（声明式）")
        sp.add_argument("--cmd"); sp.add_argument("--name")
        sp.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
        sp.add_argument("--est-hours", type=float, default=1.0)
        sp.add_argument("--kaggle-dir"); sp.add_argument("--slug")
        sp.add_argument("--require-a100", action="store_true")
        sp.add_argument("--ckpt-dataset", default=None,
                        help="跨会话/主机 checkpoint 数据集 id（harvest 推送、launch 挂载）")
        sp.add_argument("--tier", choices=["auto"] + list(TIERS), default="auto")
        sp.add_argument("--dry-run", action="store_true")

    for name in ("plan", "run"):
        sp = sub.add_parser(name); add_common(sp)
    sp = sub.add_parser("status"); add_common(sp)
    sp = sub.add_parser("harvest"); add_common(sp)
    sp.add_argument("--dest", default=None, help="覆盖 job 的 harvest_dest")
    a = p.parse_args()

    if a.job:
        try:
            job = Job.from_json(a.job)
        except Exception as e:
            print(f"作业规格解析失败 {a.job}: {e}", file=sys.stderr); return 2
        if a.slug:
            job.slug = a.slug
    elif a.action in ("status", "harvest"):
        job = Job(name="ops", cmd="", slug=a.slug, kaggle_dir=a.kaggle_dir)
    else:
        if not a.cmd:
            print("缺 --cmd 或 --job", file=sys.stderr); return 2
        job = Job(name=a.name or "unnamed", cmd=a.cmd, device=a.device,
                  est_hours=a.est_hours, require_a100=a.require_a100,
                  kaggle_dir=a.kaggle_dir, slug=a.slug,
                  ckpt_dataset=a.ckpt_dataset)
    job.ssh_host = getattr(A100SSHBackend, "ssh_host_default", "m1-a100")

    backend = decide(job, a.tier)
    problems = backend.validate(job) if a.action in ("plan", "run") else []

    if a.action == "plan":
        print(f"[plan] job={job.name} -> {backend.name}（{backend.describe()}）")
        for pr in problems:
            print(f"  ✗ {pr}", file=sys.stderr)
        backend.launch(job, dry=True)
        return 0 if not problems else 2

    if problems:
        for pr in problems:
            print(f"[{backend.name}] ✗ {pr}", file=sys.stderr)
        return 2

    if a.action == "run":
        return backend.launch(job, dry=a.dry_run)
    if a.action == "status":
        print(backend.status(job))
        return 0
    if a.action == "harvest":
        return backend.harvest(job, a.dest or job.harvest_dest, dry=False)
    return 2


if __name__ == "__main__":
    sys.exit(main())
