#!/usr/bin/env python3
"""Limit and access tests T01-T11 (queue_design.md section 12).

Run on the controller as root, with both GPU nodes idle:

    sudo python3 /opt/team7/cluster/tests/test_limits.py
    sudo python3 /opt/team7/cluster/tests/test_limits.py --only T07,T08 --evidence /root/evidence/run2

Every command, user and output is saved under the evidence directory, plus
summary.md. Exit code 1 if any test fails. T07/T08 change limits temporarily;
the script restores them from cluster.toml (install.sh accounts) at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import CLUSTER_DIR, PREFIX, Evidence, Runner, fmt_secs, load_cfg, now, require_root, summary

import team7config

RESULTS: list[tuple[str, str, str, str]] = []


def result(tid: str, title: str, passed: bool | None, observed: str) -> None:
    res = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    RESULTS.append((tid, title, res, observed.replace("|", "/")))
    print(f"    {res}: {observed}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="cluster TOML (default /etc/team7/cluster.toml)")
    ap.add_argument("--evidence", type=Path, help="evidence directory")
    ap.add_argument("--only", help="comma-separated test IDs, e.g. T03,T07")
    ap.add_argument("--force", action="store_true", help="run even if other jobs are queued")
    args = ap.parse_args()
    require_root()
    cfg = load_cfg(args.config)
    evdir = args.evidence or Path.cwd() / "evidence" / now().replace(":", "")
    ev = Evidence(evdir)
    r = Runner(ev, cfg)
    only = {t.strip().upper() for t in args.only.split(",")} if args.only else None

    def want(tid: str) -> bool:
        return only is None or tid in only

    cluster = cfg["cluster"]["name"]
    tr, evl = cfg["training"], cfg["eval"]
    tr_acct, ev_acct = tr["account"], evl["account"]
    tr_users = [u["name"] for u in cfg.users_in(tr_acct)]
    ev_users = [u["name"] for u in cfg.users_in(ev_acct)]
    if not tr_users or not ev_users:
        sys.exit(f"need at least one user in account {tr_acct!r} and one in {ev_acct!r}")
    tu1 = tr_users[0]
    tu2 = tr_users[1] if len(tr_users) > 1 else tr_users[0]
    eu1 = ev_users[0]
    debug_qos = cfg["debug"]["qos"]

    others = subprocess.run(["squeue", "-h", "-o", "%i %j %u %T"], capture_output=True, text=True).stdout
    others = [line for line in others.splitlines() if f" {PREFIX}-" not in line]
    if others and not args.force:
        sys.exit("other jobs are queued or running; results would be unreliable (use --force):\n"
                 + "\n".join(others))

    small = ["-N1", "-n1", "-c1", "--mem=100M"]

    # ---- T01 nodes ----------------------------------------------------------
    if want("T01"):
        ev.start("T01", "nodes registered with expected GRES")
        cp = r.sh(["sinfo", "-N", "-h", "-o", "%N|%t|%G"])
        seen = {line.split("|")[0]: line.split("|") for line in cp.stdout.splitlines() if "|" in line}
        bad = []
        for n in cfg["nodes"]:
            row = seen.get(n["name"])
            if not row:
                bad.append(f"{n['name']} missing")
            elif not row[1].startswith(("idle", "mix", "alloc")):
                bad.append(f"{n['name']} {row[1]}")
            elif n["gpus"] and f"gpu:{n['gpu_type']}:{n['gpus']}" not in row[2]:
                bad.append(f"{n['name']} gres {row[2]}")
        r.sh(["slurmctld", "-V"])  # the controller has no slurmd; only GPU nodes do
        result("T01", "Nodes registered", not bad, "; ".join(bad) or "all nodes idle with expected GRES")

    # ---- T02 associations -------------------------------------------------------
    if want("T02"):
        ev.start("T02", "associations match cluster.toml")
        cp = r.sh(["sacctmgr", "-n", "-P", "show", "assoc", "where", f"cluster={cluster}",
                   "format=account,user,qos,grptres"])
        rows = [line.split("|") for line in cp.stdout.splitlines()]
        bad = []
        for u in cfg["users"]:
            a = cfg.account(u["account"])
            row = next((x for x in rows if x[0] == u["account"] and x[1] == u["name"]), None)
            if not row:
                bad.append(f"{u['name']} not in {u['account']}")
            elif a["qos"] and set(row[2].split(",")) != set(a["qos"]):
                bad.append(f"{u['name']} qos {row[2]} != {','.join(a['qos'])}")
        for a in cfg["accounts"]:
            row = next((x for x in rows if x[0] == a["name"] and not x[1]), None)
            for k, v in a["grp_tres"].items():
                if not row or f"{k}={v}" not in row[3]:
                    bad.append(f"account {a['name']} GrpTRES lacks {k}={v}")
        r.sh(["sacctmgr", "show", "qos", "format=name,priority,maxwall,maxtrespj%30,maxjobspu,maxjobspa,maxsubmitpu"])
        result("T02", "Associations", not bad, "; ".join(bad) or "users, QOS lists and GrpTRES match")

    # ---- T03 / T04 jobs run on the right node -----------------------------------------
    for tid, user, sec, title in (("T03", tu1, tr, "Train user runs on train partition"),
                                  ("T04", eu1, evl, "Eval user runs on eval partition")):
        if not want(tid):
            continue
        ev.start(tid, title)
        part = cfg.partition(sec["partition"])
        gres = ["--gres=gpu:1"] if any(cfg.node(n)["gpus"] for n in part["nodes"]) else []
        cp, job = r.submit(user, tid, ["-p", sec["partition"], "-A", sec["account"], "-q", sec["qos"],
                                       "-t", "00:02:00", *small, *gres], wrap="hostname; nvidia-smi -L || true")
        if not job:
            result(tid, title, False, f"sbatch failed: {cp.stderr.strip()}")
            continue
        info = r.wait_state(job, {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}, 180)
        okk = info.get("JobState") == "COMPLETED" and info.get("NodeList") in part["nodes"]
        result(tid, title, okk, f"job {job} {info.get('JobState')} on {info.get('NodeList')}")
        r.cancel_all()

    # ---- T05 / T06 access is refused -----------------------------------------------
    for tid, user, args_, title in (
        ("T05", tu1, ["-p", tr["partition"], "-A", ev_acct], f"{tu1} cannot use account {ev_acct}"),
        ("T06", eu1, ["-p", tr["partition"], "-A", ev_acct, "-q", evl["qos"]],
         f"{eu1} cannot use partition {tr['partition']}"),
    ):
        if not want(tid):
            continue
        ev.start(tid, title)
        cp, job = r.submit(user, tid, [*args_, "-t", "00:01:00", *small], wrap="true")
        if job and tid == "T06":
            # Slurm accepts a job whose QOS is not in the partition's AllowQos, but never
            # schedules it. Pass only if it stays PENDING with that reason and gets no node.
            time.sleep(15)
            info = r.job(job)
            reason = info.get("Reason", "")
            blocked = (info.get("JobState") == "PENDING" and "not_permitted" in reason.lower()
                       and info.get("NodeList") in ("", "(null)", None))
            result(tid, title, blocked,
                   f"accepted as job {job} but blocked: {info.get('JobState')} ({reason}), "
                   f"node={info.get('NodeList') or 'none'}" if blocked else
                   f"accepted as job {job} and not blocked: {info.get('JobState')} ({reason}), "
                   f"node={info.get('NodeList') or 'none'}")
        elif job:
            info = r.job(job)
            result(tid, title, False, f"accepted as job {job} ({info.get('JobState')} {info.get('Reason')})")
        else:
            result(tid, title, True, f"rejected: {cp.stderr.strip().splitlines()[-1] if cp.stderr.strip() else cp.returncode}")
        r.cancel_all()

    # ---- T07 / T08 GPU limits on the test partition ----------------------------------
    lp, lq = cfg["tests"]["limits_partition"], cfg["tests"]["limits_qos"]
    if want("T07") or want("T07B") or want("T08"):
        try:
            part = cfg.partition(lp)
            cfg.qos(lq)
        except KeyError:
            part = None
        if part is None or len(part["nodes"]) < 2:
            for tid in ("T07", "T07b", "T08"):
                result(tid, "GPU limits", None, f"needs [tests].limits_partition {lp!r} with 2 nodes and QOS {lq!r}")
        else:
            try:
                _gpu_limit_tests(cfg, r, ev, tu1, tu2, tr_acct, lp, lq, debug_qos, want)
            finally:
                r.cancel_all()
                ev.note("restoring partition state and accounting from cluster.toml")
                r.sh(["scontrol", "update", f"PartitionName={lp}", f"State={part['state']}"])
                r.sh([str(CLUSTER_DIR / "install.sh"), "--config", str(cfg.path), "accounts"], timeout=300)

    # ---- T09 MaxWall ----------------------------------------------------------------
    if want("T09"):
        ev.start("T09", "job over MaxWall is rejected or held")
        q = cfg.qos(tr["qos"])
        if not q["max_wall"]:
            result("T09", "MaxWall", None, f"QOS {tr['qos']} has no max_wall")
        else:
            too_long = fmt_secs(team7config.parse_slurm_time(q["max_wall"]) + 3600)
            cp, job = r.submit(tu1, "T09", ["-p", tr["partition"], "-A", tr_acct, "-q", tr["qos"],
                                            "-t", too_long, *small], wrap="sleep 10")
            if not job:
                result("T09", "MaxWall", True, f"rejected at submit: {cp.stderr.strip()}")
            else:
                info = r.wait_state(job, {"RUNNING"}, 15)
                reason = info.get("Reason", "")
                okk = info.get("JobState") == "PENDING" and ("Wall" in reason or "TimeLimit" in reason)
                result("T09", "MaxWall", okk, f"--time={too_long}: {info.get('JobState')} ({reason})")
            r.cancel_all()

    # ---- T10 max jobs per account ------------------------------------------------------
    if want("T10"):
        ev.start("T10", "second train-QOS job waits on the QOS job limit")
        q = cfg.qos(tr["qos"])
        limit = q["max_jobs_pa"] if q["max_jobs_pa"] is not None else q["max_jobs_pu"]
        if limit != 1:
            result("T10", "Max jobs", None, f"QOS {tr['qos']} max_jobs_pa/pu is {limit}, test expects 1")
        else:
            base = ["-p", tr["partition"], "-A", tr_acct, "-q", tr["qos"], "-t", "00:03:00", *small]
            _, j1 = r.submit(tu1, "T10", base, wrap="sleep 120")
            i1 = r.wait_state(j1, {"RUNNING"}, 60) if j1 else {}
            _, j2 = r.submit(tu2, "T10", base, wrap="sleep 120")
            time.sleep(10)
            i2 = r.wait_state(j2, {"RUNNING"}, 5) if j2 else {}
            reason = i2.get("Reason", "")
            okk = i1.get("JobState") == "RUNNING" and i2.get("JobState") == "PENDING" and "MaxJobs" in reason
            result("T10", "Max jobs", okk, f"first {i1.get('JobState')}, second {i2.get('JobState')} ({reason})")
            r.cancel_all()

    # ---- T11 cross-user cancel ------------------------------------------------------------
    if want("T11"):
        ev.start("T11", "another user cannot cancel your job")
        other = tu2 if tu2 != tu1 else eu1
        _, job = r.submit(tu1, "T11", ["-p", tr["partition"], "-A", tr_acct, "-q", debug_qos,
                                       "-t", "00:03:00", *small], wrap="sleep 120")
        info = r.wait_state(job, {"RUNNING"}, 60) if job else {}
        if info.get("JobState") != "RUNNING":
            result("T11", "Cancel isolation", False, f"job did not start ({info.get('Reason')})")
        else:
            cp = r.sh(["scancel", job], user=other)
            time.sleep(3)
            still = r.job(job).get("JobState")
            r.sh(["scancel", job], user=tu1)
            final = r.wait_state(job, {"CANCELLED"}, 30).get("JobState")
            okk = still == "RUNNING" and final == "CANCELLED"
            result("T11", "Cancel isolation", okk,
                   f"{other}: rc={cp.returncode}, job still {still}; owner cancel -> {final}")
        r.cancel_all()

    r.sh(["scontrol", "show", "assoc_mgr", "flags=assoc"])
    r.sh(["sshare", "-l"])
    return summary(evdir, RESULTS)


def _gpu_limit_tests(cfg, r: Runner, ev: Evidence, tu1, tu2, acct, lp, lq, debug_qos, want) -> None:
    cluster = cfg["cluster"]["name"]
    ev.start("T07", "GrpTRES gres/gpu=1 blocks a second 1-GPU job while a GPU is idle")
    r.sh(["scontrol", "update", f"PartitionName={lp}", "State=UP"])
    r.sh(["sacctmgr", "-i", "modify", "account", "where", f"name={acct}", f"cluster={cluster}",
          "set", f"QOS+={lq}", "GrpTRES=gres/gpu=1"])
    r.sh(["sacctmgr", "-i", "modify", "user", "where", f"account={acct}", f"cluster={cluster}",
          "set", f"QOS+={lq}"])
    time.sleep(5)
    base = ["-p", lp, "-A", acct, "-q", lq, "-N1", "-n1", "-c1", "--mem=100M", "--gres=gpu:1", "-t", "00:05:00"]
    cp1, j1 = r.submit(tu1, "T07", base, wrap="sleep 300")
    i1 = r.wait_state(j1, {"RUNNING"}, 60) if j1 else {}
    cp2, j2 = r.submit(tu2, "T07", base, wrap="sleep 300")
    time.sleep(15)
    i2 = r.wait_state(j2, {"RUNNING"}, 5) if j2 else {}
    r.sh(["sinfo", "-N", "-p", lp, "-o", "%N %t %G"])
    reason = i2.get("Reason", "")
    okk = i1.get("JobState") == "RUNNING" and i2.get("JobState") == "PENDING" and reason.startswith("AssocGrpGRES")
    if want("T07"):
        result("T07", "GPU limit (GrpTRES)", okk,
               f"first {i1.get('JobState')} on {i1.get('NodeList')}, second {i2.get('JobState')} ({reason or cp2.stderr.strip()})")

    ev.start("T07b", "raising GrpTRES to 2 lets the second job start")
    r.sh(["sacctmgr", "-i", "modify", "account", "where", f"name={acct}", f"cluster={cluster}",
          "set", "GrpTRES=gres/gpu=2"])
    i2 = r.wait_state(j2, {"RUNNING"}, 90) if j2 else {}
    if want("T07") or want("T07B"):
        result("T07b", "GPU limit control", i2.get("JobState") == "RUNNING",
               f"second job {i2.get('JobState')} on {i2.get('NodeList')}")
    r.cancel_all()

    if want("T08"):
        ev.start("T08", "debug QOS MaxTRESPerJob gres/gpu=1 holds a 2-GPU job")
        cp, j = r.submit(tu1, "T08", ["-p", lp, "-A", acct, "-q", debug_qos, "-N2", "--ntasks-per-node=1",
                                      "-c1", "--mem=100M", "--gres=gpu:1", "-t", "00:05:00"], wrap="sleep 60")
        if not j:
            result("T08", "Per-job GPU limit", True, f"rejected at submit: {cp.stderr.strip()}")
        else:
            time.sleep(10)
            info = r.wait_state(j, {"RUNNING"}, 5)
            reason = info.get("Reason", "")
            result("T08", "Per-job GPU limit", info.get("JobState") == "PENDING" and "GRES" in reason.upper(),
                   f"{info.get('JobState')} ({reason})")
        r.cancel_all()


if __name__ == "__main__":
    sys.exit(main())
