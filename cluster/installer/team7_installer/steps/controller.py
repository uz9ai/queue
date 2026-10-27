"""Controller-only steps: Slurm config files, MariaDB, slurmdbd, slurmctld."""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from .. import render
from ..util import (
    ETC,
    Ctx,
    ensure_dir,
    header,
    info,
    is_active,
    ok,
    out,
    read_text_or_none,
    restart,
    run,
    wait_for,
    warn,
    write_file,
)

DB_PASSWORD_FILE = ETC / "secrets" / "db_password"

# Lines whose change needs a slurmctld restart (and usually slurmd restarts),
# not just `scontrol reconfigure`.
_RESTART_KEYS = re.compile(
    r"^(NodeName|SlurmctldHost|SlurmctldParameters|AuthType|CredType|SlurmUser|SlurmctldPort|"
    r"SlurmdPort|StateSaveLocation|GresTypes|SelectType|SelectTypeParameters|ProctrackType|"
    r"TaskPlugin|SchedulerType|PriorityType|AccountingStorage\w*|JobAcctGatherType|PreemptType|"
    r"ClusterName)="
)


def _changed_keys(old: str, new: str) -> set[str]:
    a = {line for line in old.splitlines() if line and not line.startswith("#")}
    b = {line for line in new.splitlines() if line and not line.startswith("#")}
    return {line.split("=", 1)[0] for line in a ^ b}


def slurm_config(ctx: Ctx) -> None:
    """Write slurm.conf, gres.conf, cgroup.conf. Records what kind of reload is needed."""
    header("Slurm config files")
    s = ctx.cfg["slurm"]
    conf_dir = Path(s["conf_dir"])
    ensure_dir(ctx, conf_dir)
    ensure_dir(ctx, s["state_save_location"], 0o700, "slurm", "slurm")
    ensure_dir(ctx, s["log_dir"], 0o755, "slurm", "slurm")

    slurm_conf = conf_dir / "slurm.conf"
    old = read_text_or_none(slurm_conf) or ""
    new = render.slurm_conf(ctx.cfg)
    changed = write_file(ctx, slurm_conf, new)
    changed |= write_file(ctx, conf_dir / "gres.conf", render.gres_conf(ctx.cfg))
    changed |= write_file(ctx, conf_dir / "cgroup.conf", render.cgroup_conf(ctx.cfg))
    keys = _changed_keys(old, new) if old else set()
    ctx.restart_ctld = bool(old) and any(_RESTART_KEYS.match(k + "=") for k in keys)
    ctx.reconfigure_ctld = changed
    ctx.node_defs_changed = "NodeName" in keys or "GresTypes" in keys
    if not changed:
        ok("Slurm config unchanged")
    elif keys:
        info("changed settings: " + ", ".join(sorted(keys)))


def database(ctx: Ctx) -> None:
    header("MariaDB for slurmdbd")
    if write_file(ctx, "/etc/mysql/mariadb.conf.d/60-team7-slurm.cnf", render.mariadb_cnf(ctx.cfg)):
        restart(ctx, "mariadb")
    elif not is_active(ctx, "mariadb"):
        restart(ctx, "mariadb")
    else:
        ok("mariadb running")

    existing = read_text_or_none(DB_PASSWORD_FILE)
    if existing:
        password = existing.strip()
    else:
        password = secrets.token_urlsafe(32)
        ensure_dir(ctx, DB_PASSWORD_FILE.parent, 0o700)
        write_file(ctx, DB_PASSWORD_FILE, password + "\n", mode=0o600, backup=False, secret=True)
        info(f"generated the database password ({DB_PASSWORD_FILE})")
    ctx.db_password = password

    d = ctx.cfg["database"]
    sql = (
        f"CREATE DATABASE IF NOT EXISTS `{d['name']}`;\n"
        f"CREATE USER IF NOT EXISTS '{d['user']}'@'localhost' IDENTIFIED BY '{password}';\n"
        f"ALTER USER '{d['user']}'@'localhost' IDENTIFIED BY '{password}';\n"
        f"GRANT ALL PRIVILEGES ON `{d['name']}`.* TO '{d['user']}'@'localhost';\n"
        "FLUSH PRIVILEGES;\n"
    )
    run(ctx, ["mysql", "--user=root"], input=sql, secret=True)
    ok(f"database {d['name']} and user {d['user']}@localhost ready")


def slurmdbd(ctx: Ctx) -> None:
    header("slurmdbd")
    s = ctx.cfg["slurm"]
    password = ctx.db_password
    if password is None:
        password = (read_text_or_none(DB_PASSWORD_FILE) or "<dry-run>").strip()
    changed = write_file(
        ctx,
        Path(s["conf_dir"]) / "slurmdbd.conf",
        render.slurmdbd_conf(ctx.cfg, password),
        mode=0o600,
        owner="slurm",
        group="slurm",
        secret=True,
    )
    if changed or not is_active(ctx, "slurmdbd") or ctx.force:
        restart(ctx, "slurmdbd")
    else:
        ok("slurmdbd running")

    def dbd_up() -> bool:
        cp = run(ctx, ["sacctmgr", "-n", "-P", "show", "cluster", "format=cluster"],
                 check=False, readonly=True, timeout=30)
        return cp.returncode == 0

    wait_for(ctx, dbd_up, "slurmdbd to answer sacctmgr", timeout=90)
    ok("slurmdbd answers")


def slurmctld(ctx: Ctx) -> None:
    header("slurmctld")
    restart_needed = ctx.restart_ctld
    reconfigure = ctx.reconfigure_ctld
    if not is_active(ctx, "slurmctld") or restart_needed or ctx.force:
        if restart_needed:
            info("restart needed: a setting changed that `scontrol reconfigure` cannot apply")
        restart(ctx, "slurmctld")
    elif reconfigure:
        run(ctx, ["scontrol", "reconfigure"])
    else:
        ok("slurmctld running, config unchanged")

    def ctld_up() -> bool:
        return "UP" in out(ctx, ["scontrol", "ping"])

    wait_for(ctx, ctld_up, "slurmctld to answer scontrol ping", timeout=90)
    ok("slurmctld answers")
    if ctx.node_defs_changed:
        warn(
            "node definitions changed. Restart slurmd on every node: "
            "`sudo systemctl restart slurmd` (or rerun install.sh node <name>)"
        )
