"""Stand-in for train.py, used by test_requeue.py (T12b).

Behaves like a real training rank from Slurm's point of view:
- writes its PID to $TEAM7_PID_DIR/rank-$RANK.pid
- on SIGUSR1, "saves a checkpoint" and exits with [slurm].requeue_exit
- after TEAM7_REQUEUE_CYCLES requeues, finishes with exit 0

Accepts and ignores any other arguments (the DeepSpeed launcher adds --local_rank).
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import team7config

cfg = team7config.load()
rank = os.environ.get("RANK", "0")
restarts = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
cycles = int(os.environ.get("TEAM7_REQUEUE_CYCLES", "3"))
pid_dir = Path(os.environ.get("TEAM7_PID_DIR") or cfg.pid_dir())
pid_dir.mkdir(parents=True, exist_ok=True)
(pid_dir / f"rank-{rank}.pid").write_text(str(os.getpid()))
print(f"dummy rank {rank}: pid {os.getpid()}, restart {restarts}/{cycles}, args {sys.argv[1:]}", flush=True)

if restarts >= cycles:
    print(f"dummy rank {rank}: done after {restarts} requeues", flush=True)
    sys.exit(0)

stop = False


def on_usr1(signum, frame):
    global stop
    stop = True


signal.signal(signal.SIGUSR1, on_usr1)
step = 0
while not stop:
    time.sleep(1)
    step += 1

print(f"dummy rank {rank}: SIGUSR1 at step {step}, saving", flush=True)
time.sleep(5)  # pretend to write a checkpoint
print(f"dummy rank {rank}: saved, exiting {cfg.requeue_exit}", flush=True)
sys.exit(cfg.requeue_exit)
