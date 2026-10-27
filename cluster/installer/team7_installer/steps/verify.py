"""Read-only checks.

check_controller / check_node run at the end of an install.
verify_cluster (install.sh verify) runs a real job on every node as a
cluster user and compares each node's cluster.toml with the controller's.
"""

from __future__ import annotations

import re

from team7config import file_sha256

from ..util import INSTALLED_TOML, Ctx, InstallError, header, info, ok, out, run, warn


def _node_states(ctx: Ctx) -> dict[str, tuple[str, str]]:
    text = out(ctx, ["sinfo", "-N", "-h", "-o", "%N|%t|%G"])
    states = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            states[parts[0]] = (parts[1], parts[2])
    return states


def check_controller(ctx: Ctx) -> None:
    header("Controller checks")
    if ctx.dry_run:
        info("[dry-run] skipped")
        return
    ping = out(ctx, ["scontrol", "ping"])
    if "UP" not in ping:
        raise InstallError(f"scontrol ping: {ping or 'no answer'}")
    ok(ping.splitlines()[0])
    states = _node_states(ctx)
    for n in ctx.cfg["nodes"]:
        st = states.get(n["name"], ("missing", ""))[0]
        info(f"{n['name']:<12} {st}")
    info(out(ctx, ["sacctmgr", "-n", "show", "assoc", "format=account%12,user%12,qos%30,grptres%20"]))
    sha = file_sha256(INSTALLED_TOML)
    print(
        f"""
Controller ready. Next, on each GPU node:

  1. Copy the munge key (only with munge_key_source = "file"), from your laptop:
       ssh ubuntu@<controller-address> 'sudo cat {ctx.cfg['secrets']['munge_key_file']}' > munge.key
       scp munge.key ubuntu@<node-address>:/tmp/munge.key      # for each node
       rm munge.key
  2. Use the same repo commit and cluster.toml (sha256 {sha[:16]}...), then run:
""" + "\n".join(
            f"       sudo ./install.sh node {n['name']} --munge-key /tmp/munge.key" for n in ctx.cfg["nodes"]
        ) + """
  3. Back here: sudo ./install.sh verify
"""
    )


def check_node(ctx: Ctx) -> None:
    header("Node checks")
    if ctx.dry_run:
        info("[dry-run] skipped")
        return
    ping = out(ctx, ["scontrol", "ping"])
    if "UP" not in ping:
        raise InstallError(f"this node cannot reach slurmctld: {ping or 'no answer'}")
    ok(ping.splitlines()[0])
    name = ctx.node["name"]
    st = _node_states(ctx).get(name)
    if st:
        ok(f"{name}: state {st[0]}, gres {st[1]}")
    print(
        f"""
Node {name} ready. On the controller run:  sudo ./install.sh verify
"""
    )


def _pick(ctx: Ctx, node: dict) -> tuple[str, str, str, str] | None:
    """(partition, account, qos, user) that can run a short job on this node."""
    debug_qos = ctx.cfg["debug"]["qos"]
    for p in ctx.cfg["partitions"]:
        if node["name"] not in p["nodes"] or p["state"] != "UP":
            continue
        for a in ctx.cfg["accounts"]:
            if p["allow_accounts"] and a["name"] not in p["allow_accounts"]:
                continue
            qos_ok = [q for q in a["qos"] if not p["allow_qos"] or q in p["allow_qos"]]
            if not qos_ok:
                continue
            users = [u for u in ctx.cfg["users"] if u["account"] == a["name"]]
            if not users:
                continue
            qos = debug_qos if debug_qos in qos_ok else qos_ok[0]
            return p["name"], a["name"], qos, users[0]["name"]
    return None


def verify_cluster(ctx: Ctx) -> int:
    header("Cluster verification")
    if ctx.dry_run:
        info("[dry-run] skipped")
        return 0
    failures = 0
    ping = out(ctx, ["scontrol", "ping"])
    if "UP" not in ping:
        raise InstallError(f"slurmctld is not up: {ping}")
    want_sha = file_sha256(INSTALLED_TOML)
    states = _node_states(ctx)
    for n in ctx.cfg["nodes"]:
        name = n["name"]
        st, gres = states.get(name, ("missing", ""))
        prefix = f"{name:<12}"
        if st in ("missing",) or st.startswith(("down", "drain", "fail", "unk")):
            warn(f"{prefix} state {st}")
            failures += 1
            continue
        if n["gpus"] and f"gpu:{n['gpu_type']}:{n['gpus']}" not in gres:
            warn(f"{prefix} gres is {gres!r}, expected gpu:{n['gpu_type']}:{n['gpus']}")
            failures += 1
        pick = _pick(ctx, n)
        if not pick:
            warn(f"{prefix} no UP partition/account/user can run a test job here")
            failures += 1
            continue
        part, acct, qos, user = pick
        body = "hostname -s; sha256sum /etc/team7/cluster.toml | cut -d' ' -f1"
        if n["gpus"]:
            body += "; nvidia-smi -L"
        cmd = ["runuser", "-u", user, "--", "srun", "--chdir=/tmp", "-p", part, "-A", acct, "-q", qos,
               "-w", name, "-N1", "-n1", "-t", "2", "--immediate=60", "-J", "team7-verify"]
        if n["gpus"]:
            cmd.append("--gres=gpu:1")
        cmd += ["bash", "-c", body]
        cp = run(ctx, cmd, check=False, timeout=180)
        lines = (cp.stdout or "").splitlines()
        if cp.returncode != 0:
            err = (cp.stderr or "").strip().splitlines()
            msg = err[-1] if err else f"rc={cp.returncode}"
            if "Unable to allocate resources" in msg or "Immediate" in msg:
                warn(f"{prefix} busy, test job could not start within 60 s ({msg})")
            else:
                warn(f"{prefix} test job failed: {msg}")
            failures += 1
            continue
        host = lines[0] if lines else ""
        sha = lines[1] if len(lines) > 1 else ""
        gpu = next((x for x in lines if x.startswith("GPU ")), "")
        if host != name:
            warn(f"{prefix} job ran on {host!r}")
            failures += 1
        if sha != want_sha:
            warn(f"{prefix} /etc/team7/cluster.toml differs from the controller's; rerun install.sh node {name}")
            failures += 1
        else:
            ok(f"{prefix} {st}, job as {user} ({acct}/{qos}) ok, cluster.toml matches"
               + (f", {re.sub(r' [(].*', '', gpu)}" if gpu else ""))
    if failures:
        warn(f"{failures} problem(s) found")
        return 1
    ok("all nodes pass")
    return 0
