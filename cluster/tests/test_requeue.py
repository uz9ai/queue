#!/usr/bin/env python3
"""Self-requeue tests (queue_design.md T12 and T12b).

T12  plain batch script: traps SIGUSR1 and exits with [slurm].requeue_exit.
T12b the real /opt/team7/jobs/train.sbatch (DeepSpeed launcher) running
     tests/dummy_train.py, which writes its PID file and exits on SIGUSR1.

Each job must be requeued [tests].requeue_cycles times and then complete.
Takes roughly cycles x [tests].requeue_time per test. Run on the controller:

    sudo python3 /opt/team7/cluster/tests/test_requeue.py            # both
    sudo python3 /opt/team7/cluster/tests/test_requeue.py --only T12b --cycles 1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from _common import Evidence, Runner, load_cfg, now, require_root, summary

import team7config

RESULTS: list[tuple[str, str, str, str]] = []

PLAIN = """#!/bin/bash
REQUEUE_EXIT={rx}
CYCLES={cycles}
echo "start restarts=${{SLURM_RESTART_COUNT:-0}} $(date -u +%FT%TZ)"
if [ "${{SLURM_RESTART_COUNT:-0}}" -ge "$CYCLES" ]; then echo done; exit 0; fi
got=0
trap 'got=1' USR1
while [ "$got" -eq 0 ]; do sleep 1; done
echo "USR1 received, exiting $REQUEUE_EXIT"
exit "$REQUEUE_EXIT"
"""


def run_cycle_test(r: Runner, ev: Evidence, tid: str, title: str, user: str, args: list[str],
                   script: str, cycles: int, budget: int) -> None:
    ev.start(tid, title)
    cp, job = r.submit(user, tid, args, script=script)
    if not job:
        RESULTS.append((tid, title, "FAIL", f"sbatch failed: {cp.stderr.strip()}"))
        return
    info = r.wait_state(job, {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}, budget)
    sacct = r.sh(["sacct", "-D", "-X", "-n", "-P", "-j", job, "--format=JobID,State,ExitCode,Start,End,NodeList"])
    requeued = sum(1 for line in sacct.stdout.splitlines() if "|REQUEUED|" in line)
    restarts = info.get("Restarts", "?")
    okk = info.get("JobState") == "COMPLETED" and str(restarts) == str(cycles)
    obs = (f"job {job}: {info.get('JobState')} exit {info.get('ExitCode')}, Restarts={restarts}, "
           f"{requeued} REQUEUED record(s) in sacct")
    RESULTS.append((tid, title, "PASS" if okk else "FAIL", obs))
    print(f"    {'PASS' if okk else 'FAIL'}: {obs}")
    r.cancel_all()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--evidence", type=Path)
    ap.add_argument("--only", help="T12 or T12b")
    ap.add_argument("--cycles", type=int, help="override [tests].requeue_cycles")
    args = ap.parse_args()
    require_root()
    cfg = load_cfg(args.config)
    t = cfg["tests"]
    cycles = args.cycles if args.cycles is not None else t["requeue_cycles"]
    tr = cfg["training"]
    users = cfg.users_in(tr["account"])
    if not users:
        sys.exit(f"no users in account {tr['account']}")
    user = users[0]["name"]
    evdir = args.evidence or Path.cwd() / "evidence" / ("requeue-" + now().replace(":", ""))
    ev = Evidence(evdir)
    r = Runner(ev, cfg)
    only = {x.strip().upper() for x in args.only.split(",")} if args.only else None
    secs = team7config.parse_slurm_time(t["requeue_time"])
    lead = t["requeue_signal_lead_s"]
    budget = cycles * (secs + 300) + secs + 300
    gpus = ["--gres=gpu:1"] if any(n["gpus"] for n in cfg.nodes_by_role("train")) else []
    common = ["-p", tr["partition"], "-A", tr["account"], "-q", tr["qos"], f"--time={t['requeue_time']}",
              f"--signal=B:USR1@{lead}", "--requeue", "--open-mode=append", *gpus]

    if only is None or "T12" in only:
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False, dir="/tmp") as f:
            f.write(PLAIN.format(rx=cfg.requeue_exit, cycles=cycles))
        Path(f.name).chmod(0o755)
        run_cycle_test(r, ev, "T12", "self-requeue, plain batch script", user,
                       [*common, "-N1", "-n1", "-c1", "--mem=200M"], f.name, cycles, budget)
        Path(f.name).unlink()

    if only is None or "T12B" in only:
        train_sbatch = Path("/opt/team7/jobs/train.sbatch")
        # Path on the GPU node (the installer copies cluster/ to /opt/team7/cluster everywhere).
        dummy = Path("/opt/team7/cluster/tests/dummy_train.py")
        if not train_sbatch.exists():
            RESULTS.append(("T12b", "self-requeue through DeepSpeed launcher", "SKIP", f"{train_sbatch} missing"))
        else:
            export = (f"--export=ALL,TEAM7_TRAIN_SCRIPT={dummy},TEAM7_DS_CONFIG=,"
                      f"TEAM7_REQUEUE_CYCLES={cycles}")
            run_cycle_test(r, ev, "T12b", "self-requeue through DeepSpeed launcher", user,
                           [*common, export], str(train_sbatch), cycles, budget)

    return summary(evdir, RESULTS)


if __name__ == "__main__":
    sys.exit(main())
