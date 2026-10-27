"""Helpers shared by the cluster tests. Run the tests on the controller as root."""

from __future__ import annotations

import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLUSTER_DIR = HERE.parent
sys.path.insert(0, str(CLUSTER_DIR / "team7config"))

import team7config  # noqa: E402

PREFIX = "t7test"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Evidence:
    """Writes every command, its user and its output to <dir>/<test>.log."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.current: Path | None = None

    def start(self, test_id: str, title: str) -> None:
        self.current = self.root / f"{test_id}.log"
        self.current.write_text(f"# {test_id}: {title}\n# started {now()}\n\n")
        print(f"\n--- {test_id}: {title}", flush=True)

    def note(self, text: str) -> None:
        print(f"    {text}", flush=True)
        if self.current:
            with self.current.open("a") as f:
                f.write(f"# {text}\n")

    def record(self, user: str, cmd: list[str], cp: subprocess.CompletedProcess) -> None:
        if not self.current:
            return
        with self.current.open("a") as f:
            f.write(f"[{now()}] {user}$ {shlex.join(cmd)}\n")
            if cp.stdout:
                f.write(cp.stdout if cp.stdout.endswith("\n") else cp.stdout + "\n")
            if cp.stderr:
                f.write("".join(f"stderr: {line}\n" for line in cp.stderr.splitlines()))
            f.write(f"exit={cp.returncode}\n\n")


class Runner:
    def __init__(self, evidence: Evidence, cfg):
        self.ev = evidence
        self.cfg = cfg
        self.jobs: list[str] = []

    def sh(self, cmd: list[str], user: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
        full = ["runuser", "-u", user, "--", *cmd] if user else cmd
        try:
            cp = subprocess.run(full, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        except subprocess.TimeoutExpired:
            cp = subprocess.CompletedProcess(full, 124, "", f"timed out after {timeout}s")
        self.ev.record(user or "root", cmd, cp)
        return cp

    def submit(self, user: str, test_id: str, args: list[str], wrap: str | None = None,
               script: str | None = None) -> tuple[subprocess.CompletedProcess, str | None]:
        logs = self.cfg["logs"]["dir"]
        cmd = ["sbatch", "--parsable", f"--job-name={PREFIX}-{test_id}", "--chdir=/tmp",
               f"--output={logs}/{PREFIX}-%j.out", *args]
        if wrap is not None:
            cmd.append(f"--wrap={wrap}")
        if script is not None:
            cmd.append(script)
        cp = self.sh(cmd, user=user)
        job = cp.stdout.strip().split(";")[0] if cp.returncode == 0 and cp.stdout.strip() else None
        if job:
            self.jobs.append(job)
        return cp, job

    def job(self, job_id: str) -> dict[str, str]:
        cp = subprocess.run(["scontrol", "-o", "show", "job", job_id], capture_output=True, text=True)
        info: dict[str, str] = {}
        if cp.returncode == 0:
            for m in re.finditer(r"(\w+)=(\S*)", cp.stdout):
                info.setdefault(m.group(1), m.group(2))
        return info

    def wait_state(self, job_id: str, states: set[str], timeout: float) -> dict[str, str]:
        deadline = time.time() + timeout
        info = {}
        while time.time() < deadline:
            info = self.job(job_id)
            if info.get("JobState") in states:
                break
            time.sleep(2)
        self.ev.note(f"job {job_id}: state={info.get('JobState')} reason={info.get('Reason')} "
                     f"node={info.get('NodeList')} restarts={info.get('Restarts')}")
        return info

    def cancel_all(self) -> None:
        if self.jobs:
            subprocess.run(["scancel", *self.jobs], capture_output=True)
            deadline = time.time() + 60
            while time.time() < deadline:
                live = [j for j in self.jobs if self.job(j).get("JobState") in
                        {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}]
                if not live:
                    break
                time.sleep(2)
            self.jobs.clear()


def require_root() -> None:
    if os.geteuid() != 0:
        sys.exit("run as root on the controller: sudo python3 " + " ".join(sys.argv))


def load_cfg(path: str | None):
    return team7config.load(path or None)


def fmt_secs(secs: int) -> str:
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def summary(evdir: Path, results: list[tuple[str, str, str, str]]) -> int:
    lines = ["| ID | Test | Result | Observed |", "|---|---|---|---|"]
    for tid, title, res, obs in results:
        lines.append(f"| {tid} | {title} | {res} | {obs} |")
    text = "\n".join(lines) + "\n"
    (evdir / "summary.md").write_text(f"# Test results ({now()})\n\n" + text)
    print("\n" + text)
    print(f"evidence: {evdir}")
    return 1 if any(r[2] == "FAIL" for r in results) else 0
