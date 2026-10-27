"""Load and validate the Team 7 cluster TOML.

The same file configures Slurm (through the installer) and the training code.
Standard library only, so the installer can use it with the system python3.

    import team7config
    cfg = team7config.load()              # $TEAM7_CONFIG or /etc/team7/cluster.toml
    cfg.requeue_exit                      # 75
    cfg.get("checkpoints.local_dir")      # "/scratch/ckpt"
    cfg.pid_dir()                         # Path("/tmp/team7-<SLURM_JOB_ID>")
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import os
import re
import socket
import tomllib
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "Config",
    "ConfigError",
    "DEFAULTS",
    "INSTALLED_PATH",
    "load",
    "load_raw",
    "normalize",
    "validate",
    "parse_slurm_time",
    "tres_string",
    "file_sha256",
]

INSTALLED_PATH = Path("/etc/team7/cluster.toml")
ENV_VAR = "TEAM7_CONFIG"

ROLES = ("train", "eval")
ADMIN_LEVELS = ("None", "Operator", "Administrator")

# ---------------------------------------------------------------------------
# Defaults. DOCS.md documents every key. Keep the two in sync.
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "cluster": {
        "name": None,
        "os_version": "24.04",
        "slurm_version": "23.11",
        "ntp_server": "169.254.169.123",
        "timezone": "",
    },
    "controller": {"name": None, "ip": None},
    "unix": {
        "group": "team7",
        "gid": 2000,
        "slurm_uid": 64030,
        "slurm_gid": 64030,
        "shell": "/bin/bash",
    },
    "secrets": {
        "munge_key_source": "file",
        "munge_key_file": "/root/team7/munge.key",
        "ssm_parameter": "/team7/munge_key",
        "ssm_region": "",
    },
    "database": {
        "name": "slurm_acct_db",
        "user": "slurm",
        "innodb_buffer_pool_size": "512M",
        "innodb_lock_wait_timeout": 900,
    },
    "slurm": {
        "conf_dir": "/etc/slurm",
        "ctld_port": 6817,
        "slurmd_port": 6818,
        "dbd_port": 6819,
        "srun_port_range": "60001-63000",
        "state_save_location": "/var/spool/slurmctld",
        "slurmd_spool_dir": "/var/spool/slurmd",
        "log_dir": "/var/log/slurm",
        "select_type_parameters": "CR_Core_Memory",
        "scheduler_type": "sched/backfill",
        "scheduler_parameters": ["nohold_on_prolog_fail"],
        "task_plugin": "task/cgroup,task/affinity",
        "preempt_type": "preempt/none",
        "accounting_enforce": ["associations", "limits", "qos", "safe"],
        "accounting_tres": ["gres/gpu"],
        "job_requeue": True,
        "requeue_exit": 75,
        "return_to_service": 1,
        "slurmctld_timeout": 120,
        "slurmd_timeout": 300,
        "kill_wait": 30,
        "min_job_age": 300,
        "priority_decay_half_life": "7-00:00:00",
        "priority_max_age": "7-00:00:00",
        "priority_weights": {
            "fairshare": 10000,
            "age": 1000,
            "qos": 2000,
            "partition": 0,
            "job_size": 0,
        },
        "cgroup": {
            "constrain_cores": True,
            "constrain_ram_space": True,
            "constrain_devices": True,
        },
        "extra": {},
    },
    "health": {
        "enabled": False,
        "launch_script": "/opt/team7/src/health/launch.sh",
        "quick_args": "--quick",
        "prolog": True,
        "periodic": True,
        "interval_s": 300,
        "node_state": "ANY",
        "check_timeout_s": 120,
        "prolog_timeout_s": 180,
        "log_dir": "/var/log/team7",
    },
    "storage": {
        "scratch": {
            "type": "nvme",
            "mount": "/scratch",
            "device": "auto",
            "filesystem": "xfs",
        },
        "shared": {
            "type": "none",
            "mount": "/shared",
            "efs_id": "",
            "region": "",
            "nfs_server": "",
            "nfs_export": "/",
            "options": "",
        },
    },
    "checkpoints": {
        "local_dir": "/scratch/ckpt",
        "durable": "s3",
        "s3_uri": "",
        "shared_dir": "",
        "keep_local": 2,
        "keep_durable_last": 3,
        "keep_durable_every": 0,
    },
    "logs": {"dir": "/scratch/logs"},
    "code": {
        "source": "none",
        "local_path": "..",
        "git_url": "",
        "git_ref": "main",
        "dest": "/opt/team7/src",
        "exclude": [".git", ".venv", "__pycache__", "*.pyc"],
    },
    "python": {
        "project": "pyenv",
        "version": "3.12",
        "uv_version": "0.13.6",
        "uv_sha256": "",
        "venv": "/opt/team7/venv",
        "python_install_dir": "/opt/team7/python",
        "cache_dir": "/var/cache/team7-uv",
        "cuda_home": "/usr/local/cuda",
        "cuda_arch_list": "7.5",
        "torch_extensions_dir": "/opt/team7/torch_extensions",
        "deepspeed_prebuild_ops": ["FusedAdamBuilder"],
    },
    "training": {
        "partition": "train",
        "account": "train",
        "qos": "train",
        "job_name": "train",
        "script": "/opt/team7/src/train.py",
        "args": [],
        "deepspeed_config": "",
        "time": "24:00:00",
        "signal_lead_s": 300,
        "pid_dir": "/tmp/team7-{job_id}",
        "gpus_per_job": 0,
        "cpus_per_task": 0,
        "mem": "0",
        "master_port_base": 20000,
        "precision": "fp16",
    },
    "eval": {
        "partition": "eval",
        "account": "eval",
        "qos": "eval",
        "job_name": "eval",
        "script": "/opt/team7/src/eval.py",
        "args": [],
        "time": "04:00:00",
        "gpus_per_job": 1,
        "cpus_per_task": 2,
        "mem": "8G",
    },
    "debug": {
        "qos": "debug",
        "time": "00:10:00",
        "gpus_per_job": 1,
        "cpus_per_task": 1,
        "mem": "4G",
    },
    "tests": {
        "limits_partition": "limits-test",
        "limits_qos": "limits_test",
        "requeue_cycles": 3,
        "requeue_time": "00:04:00",
        "requeue_signal_lead_s": 60,
    },
}

LIST_DEFAULTS: dict[str, dict[str, Any]] = {
    "nodes": {
        "name": None,
        "role": None,
        "ip": None,
        "gpus": 1,
        "gpu_type": "",
        "gpu_name_match": "",
        "cpus": None,
        "sockets": None,
        "cores_per_socket": None,
        "threads_per_core": None,
        "real_memory_mb": None,
        "features": [],
        "weight": 1,
    },
    "partitions": {
        "name": None,
        "nodes": [],
        "allow_accounts": [],
        "allow_qos": [],
        "max_time": "UNLIMITED",
        "default_time": "",
        "state": "UP",
        "default": False,
        "extra": {},
    },
    "qos": {
        "name": None,
        "description": "",
        "priority": 0,
        "max_wall": "",
        "max_tres_per_job": {},
        "max_tres_per_user": {},
        "max_jobs_pu": None,
        "max_jobs_pa": None,
        "max_submit_jobs_pu": None,
        "max_submit_jobs_pa": None,
        "flags": [],
    },
    "accounts": {
        "name": None,
        "description": "",
        "organization": "",
        "fairshare": 1,
        "grp_tres": {},
        "grp_jobs": None,
        "max_jobs": None,
        "qos": [],
        "default_qos": "",
    },
    "users": {
        "name": None,
        "uid": None,
        "account": None,
        "admin_level": "None",
        "fairshare": 1,
        "ssh_keys": [],
        "groups": [],
    },
}


class ConfigError(Exception):
    """Raised when the TOML cannot be loaded or fails validation."""

    def __init__(self, errors: list[str], path: str | os.PathLike | None = None):
        self.errors = errors
        where = f" in {path}" if path else ""
        super().__init__(f"{len(errors)} config error(s){where}:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"^(?:(?P<d>\d+)-(?P<dh>\d{1,2})(?::(?P<dm>\d{1,2})(?::(?P<ds>\d{1,2}))?)?"
    r"|(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2})"
    r"|(?P<m2>\d+):(?P<s2>\d{1,2})"
    r"|(?P<m3>\d+))$"
)


def parse_slurm_time(value: str) -> int | None:
    """Slurm time string to seconds. Returns None for UNLIMITED/INFINITE.

    Accepts the Slurm forms: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S.
    Raises ValueError on anything else.
    """
    v = str(value).strip()
    if v.upper() in ("UNLIMITED", "INFINITE"):
        return None
    m = _TIME_RE.match(v)
    if not m:
        raise ValueError(f"not a Slurm time: {value!r}")
    g = m.groupdict()
    if g["d"] is not None:
        return (
            int(g["d"]) * 86400
            + int(g["dh"]) * 3600
            + int(g["dm"] or 0) * 60
            + int(g["ds"] or 0)
        )
    if g["h"] is not None:
        return int(g["h"]) * 3600 + int(g["m"]) * 60 + int(g["s"])
    if g["m2"] is not None:
        return int(g["m2"]) * 60 + int(g["s2"])
    return int(g["m3"]) * 60


def tres_string(tres: Mapping[str, Any]) -> str:
    """{"gres/gpu": 1, "cpu": 4} -> "cpu=4,gres/gpu=1"."""
    return ",".join(f"{k}={v}" for k, v in sorted(tres.items()))


def file_sha256(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping) and out[k] is not None:
            # "extra"-style free-form tables: replace, don't merge with defaults.
            out[k] = _deep_merge(out[k], v) if out[k] else dict(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def normalize(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults. Does not validate."""
    cfg = _deep_merge(DEFAULTS, {k: v for k, v in raw.items() if k not in LIST_DEFAULTS})
    for key, item_defaults in LIST_DEFAULTS.items():
        items = raw.get(key, [])
        if not isinstance(items, list):
            cfg[key] = items  # validate() reports the type error
            continue
        cfg[key] = [
            _deep_merge(item_defaults, it) if isinstance(it, Mapping) else it for it in items
        ]
    return cfg


def load_raw(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_HOST_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_TRES_KEY_RE = re.compile(r"^(cpu|mem|node|energy|billing|gres/[a-z0-9_]+(:[a-z0-9_]+)?)$")


def validate(cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a normalized config."""
    e: list[str] = []
    w: list[str] = []

    def req(section: str, key: str):
        if cfg.get(section, {}).get(key) in (None, ""):
            e.append(f"[{section}].{key} is required")

    def check_time(where: str, value, allow_empty=False):
        if value in ("", None):
            if not allow_empty:
                e.append(f"{where} is required")
            return None
        try:
            return parse_slurm_time(value)
        except ValueError as ex:
            e.append(f"{where}: {ex}")
            return None

    def check_abs(where: str, value):
        if not isinstance(value, str) or not value.startswith("/"):
            e.append(f"{where} must be an absolute path, got {value!r}")

    def check_tres(where: str, tres):
        if not isinstance(tres, Mapping):
            e.append(f"{where} must be a table like {{ \"gres/gpu\" = 1 }}")
            return
        for k, v in tres.items():
            if not _TRES_KEY_RE.match(k):
                e.append(f"{where}: unknown TRES {k!r} (use cpu, mem, node, gres/gpu, ...)")
            if not isinstance(v, (int, str)) or (isinstance(v, int) and v < -1):
                e.append(f"{where}.{k}: must be an integer >= -1 or a size string")

    # -- cluster / controller ----------------------------------------------
    req("cluster", "name")
    name = cfg["cluster"].get("name")
    if name and not _NAME_RE.match(str(name)):
        e.append(f"[cluster].name {name!r}: lowercase letters, digits, '-' or '_', max 32 chars")
    req("controller", "name")
    req("controller", "ip")

    hosts: dict[str, str] = {}
    ips: dict[str, str] = {}

    def add_host(where: str, hname, ip):
        if hname:
            if not _HOST_RE.match(str(hname)):
                e.append(f"{where}.name {hname!r} is not a valid short hostname")
            if hname in hosts:
                e.append(f"{where}.name {hname!r} duplicates {hosts[hname]}")
            hosts[hname] = where
        if ip:
            try:
                ipaddress.IPv4Address(ip)
            except ValueError:
                e.append(f"{where}.ip {ip!r} is not an IPv4 address")
            if ip in ips:
                e.append(f"{where}.ip {ip} duplicates {ips[ip]}")
            ips[ip] = where

    add_host("[controller]", cfg["controller"].get("name"), cfg["controller"].get("ip"))

    # -- nodes ---------------------------------------------------------------
    nodes = cfg.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        e.append("at least one [[nodes]] entry is required")
        nodes = []
    node_names = []
    for i, n in enumerate(nodes):
        where = f"[[nodes]][{i}]"
        if not isinstance(n, Mapping):
            e.append(f"{where} must be a table")
            continue
        for k in ("name", "role", "ip", "cpus", "real_memory_mb"):
            if n.get(k) in (None, ""):
                e.append(f"{where}.{k} is required")
        add_host(where, n.get("name"), n.get("ip"))
        node_names.append(n.get("name"))
        if n.get("role") not in (None, "") and n["role"] not in ROLES:
            e.append(f"{where}.role must be one of {ROLES}, got {n['role']!r}")
        if not isinstance(n.get("gpus"), int) or n["gpus"] < 0:
            e.append(f"{where}.gpus must be an integer >= 0")
        elif n["gpus"] > 0 and not n.get("gpu_type"):
            e.append(f"{where}.gpu_type is required when gpus > 0 (e.g. \"t4\")")
        if n.get("gpu_type") and not re.match(r"^[a-z0-9_]+$", n["gpu_type"]):
            e.append(f"{where}.gpu_type {n['gpu_type']!r}: lowercase letters, digits, '_'")
        topo = [n.get("sockets"), n.get("cores_per_socket"), n.get("threads_per_core")]
        if any(t is not None for t in topo) and not all(t is not None for t in topo):
            e.append(f"{where}: set all of sockets, cores_per_socket, threads_per_core, or none")
        elif all(isinstance(t, int) for t in topo) and isinstance(n.get("cpus"), int):
            if topo[0] * topo[1] * topo[2] != n["cpus"]:
                e.append(
                    f"{where}: sockets*cores_per_socket*threads_per_core = "
                    f"{topo[0] * topo[1] * topo[2]} but cpus = {n['cpus']}"
                )
        elif n.get("cpus") is not None and all(t is None for t in topo):
            w.append(f"{where}: no CPU topology given; copy it from `slurmd -C` to avoid drains")

    # -- unix users / slurm user --------------------------------------------
    for k in ("gid", "slurm_uid", "slurm_gid"):
        if not isinstance(cfg["unix"].get(k), int):
            e.append(f"[unix].{k} must be an integer")

    # -- qos -----------------------------------------------------------------
    qos_list = cfg.get("qos") if isinstance(cfg.get("qos"), list) else []
    qos_by_name: dict[str, Mapping] = {}
    for i, q in enumerate(qos_list):
        where = f"[[qos]][{i}]"
        qn = q.get("name")
        if not qn or not _NAME_RE.match(str(qn)):
            e.append(f"{where}.name {qn!r} is missing or invalid")
            continue
        if qn in qos_by_name:
            e.append(f"{where}.name {qn!r} is defined twice")
        qos_by_name[qn] = q
        check_time(f"{where}.max_wall", q.get("max_wall"), allow_empty=True)
        check_tres(f"{where}.max_tres_per_job", q.get("max_tres_per_job"))
        check_tres(f"{where}.max_tres_per_user", q.get("max_tres_per_user"))
        for k in ("max_jobs_pu", "max_jobs_pa", "max_submit_jobs_pu", "max_submit_jobs_pa"):
            v = q.get(k)
            if v is not None and (not isinstance(v, int) or v < -1):
                e.append(f"{where}.{k} must be an integer >= -1")
        if not isinstance(q.get("priority"), int) or q["priority"] < 0:
            e.append(f"{where}.priority must be an integer >= 0")

    # -- accounts --------------------------------------------------------------
    acct_by_name: dict[str, Mapping] = {}
    for i, a in enumerate(cfg.get("accounts") or []):
        where = f"[[accounts]][{i}]"
        an = a.get("name")
        if not an or not _NAME_RE.match(str(an)):
            e.append(f"{where}.name {an!r} is missing or invalid")
            continue
        if an in ("root",):
            e.append(f"{where}.name 'root' is reserved by Slurm")
        if an in acct_by_name:
            e.append(f"{where}.name {an!r} is defined twice")
        acct_by_name[an] = a
        check_tres(f"{where}.grp_tres", a.get("grp_tres"))
        for qn in a.get("qos") or []:
            if qn not in qos_by_name:
                e.append(f"{where}.qos: unknown QOS {qn!r}")
        dq = a.get("default_qos")
        if dq and dq not in (a.get("qos") or []):
            e.append(f"{where}.default_qos {dq!r} must be in its qos list")
        if not a.get("qos"):
            w.append(f"{where}: empty qos list, users get Slurm's default QOS 'normal'")

    if not acct_by_name:
        e.append("at least one [[accounts]] entry is required")

    # -- users -------------------------------------------------------------------
    uids: dict[int, str] = {}
    user_names: set[str] = set()
    for i, u in enumerate(cfg.get("users") or []):
        where = f"[[users]][{i}]"
        un = u.get("name")
        if not un or not re.match(r"^[a-z_][a-z0-9_-]{0,31}$", str(un)):
            e.append(f"{where}.name {un!r} is missing or not a valid Unix user name")
        elif un in user_names:
            e.append(f"{where}.name {un!r} is defined twice")
        else:
            user_names.add(un)
        uid = u.get("uid")
        if not isinstance(uid, int) or uid < 1000:
            e.append(f"{where}.uid must be an integer >= 1000")
        elif uid in uids:
            e.append(f"{where}.uid {uid} duplicates user {uids[uid]}")
        elif uid == cfg["unix"].get("slurm_uid"):
            e.append(f"{where}.uid {uid} clashes with [unix].slurm_uid")
        else:
            uids[uid] = un
        if u.get("account") not in acct_by_name:
            e.append(f"{where}.account {u.get('account')!r} is not a defined account")
        if u.get("admin_level") not in ADMIN_LEVELS:
            e.append(f"{where}.admin_level must be one of {ADMIN_LEVELS}")
        if not isinstance(u.get("ssh_keys"), list):
            e.append(f"{where}.ssh_keys must be a list of public key strings")

    # -- partitions --------------------------------------------------------------
    part_by_name: dict[str, Mapping] = {}
    defaults = 0
    for i, p in enumerate(cfg.get("partitions") or []):
        where = f"[[partitions]][{i}]"
        pn = p.get("name")
        if not pn or not _NAME_RE.match(str(pn)):
            e.append(f"{where}.name {pn!r} is missing or invalid")
            continue
        if pn in part_by_name:
            e.append(f"{where}.name {pn!r} is defined twice")
        part_by_name[pn] = p
        if not p.get("nodes"):
            e.append(f"{where}.nodes must list at least one node")
        for nn in p.get("nodes") or []:
            if nn not in node_names:
                e.append(f"{where}.nodes: unknown node {nn!r}")
        for an in p.get("allow_accounts") or []:
            if an not in acct_by_name:
                e.append(f"{where}.allow_accounts: unknown account {an!r}")
        for qn in p.get("allow_qos") or []:
            if qn not in qos_by_name:
                e.append(f"{where}.allow_qos: unknown QOS {qn!r}")
        mt = check_time(f"{where}.max_time", p.get("max_time"))
        dt = check_time(f"{where}.default_time", p.get("default_time"), allow_empty=True)
        if mt is not None and dt is not None and dt > mt:
            e.append(f"{where}.default_time is longer than max_time")
        if p.get("state") not in ("UP", "DOWN", "DRAIN", "INACTIVE"):
            e.append(f"{where}.state must be UP, DOWN, DRAIN or INACTIVE")
        if p.get("default"):
            defaults += 1
        for qn in p.get("allow_qos") or []:
            q = qos_by_name.get(qn)
            if q and q.get("max_wall") and mt is not None:
                try:
                    qw = parse_slurm_time(q["max_wall"])
                    if qw is not None and qw > mt:
                        w.append(
                            f"{where}: QOS {qn!r} max_wall ({q['max_wall']}) is longer than "
                            f"partition max_time ({p['max_time']}); the partition limit wins"
                        )
                except ValueError:
                    pass
    if defaults > 1:
        e.append("only one partition can have default = true")
    for nn in node_names:
        if nn and not any(
            nn in p.get("nodes", []) and p.get("state") == "UP" for p in part_by_name.values()
        ):
            w.append(f"node {nn!r} is not in any UP partition; no job can run on it")

    # -- job templates (training / eval / debug) ----------------------------------
    def check_template(sec: str, role: str | None):
        t = cfg[sec]
        pn = t.get("partition")
        if pn is not None and pn not in part_by_name:
            e.append(f"[{sec}].partition {pn!r} is not a defined partition")
        if "account" in t and t["account"] not in acct_by_name:
            e.append(f"[{sec}].account {t['account']!r} is not a defined account")
        qn = t.get("qos")
        if qn not in qos_by_name:
            e.append(f"[{sec}].qos {qn!r} is not a defined QOS")
        a = acct_by_name.get(t.get("account"))
        if a and qn and qn not in (a.get("qos") or []):
            e.append(f"[{sec}].qos {qn!r} is not in account {a['name']!r}'s qos list")
        p = part_by_name.get(pn)
        if p:
            if a and p.get("allow_accounts") and a["name"] not in p["allow_accounts"]:
                e.append(f"[{sec}]: account {a['name']!r} is not allowed in partition {pn!r}")
            if qn and p.get("allow_qos") and qn not in p["allow_qos"]:
                e.append(f"[{sec}]: QOS {qn!r} is not allowed in partition {pn!r}")
            if role:
                for nn in p.get("nodes") or []:
                    n = next((x for x in nodes if isinstance(x, Mapping) and x.get("name") == nn), None)
                    if n and n.get("role") != role:
                        w.append(f"[{sec}].partition {pn!r} includes node {nn!r} with role {n.get('role')!r}")
        secs = check_time(f"[{sec}].time", t.get("time"))
        if secs is not None:
            q = qos_by_name.get(qn)
            if q and q.get("max_wall"):
                try:
                    qw = parse_slurm_time(q["max_wall"])
                    if qw is not None and secs > qw:
                        e.append(f"[{sec}].time {t['time']} exceeds QOS {qn!r} max_wall {q['max_wall']}")
                except ValueError:
                    pass
            if p:
                try:
                    pm = parse_slurm_time(p.get("max_time", "UNLIMITED"))
                    if pm is not None and secs > pm:
                        e.append(f"[{sec}].time {t['time']} exceeds partition {pn!r} max_time")
                except ValueError:
                    pass
        return secs

    train_secs = check_template("training", "train")
    check_template("eval", "eval")
    # debug has no fixed partition/account: users pass -p and -A.
    if cfg["debug"].get("qos") not in qos_by_name:
        e.append(f"[debug].qos {cfg['debug'].get('qos')!r} is not a defined QOS")
    check_time("[debug].time", cfg["debug"].get("time"))

    tr = cfg["training"]
    lead = tr.get("signal_lead_s")
    if not isinstance(lead, int) or lead < 30:
        e.append("[training].signal_lead_s must be an integer >= 30")
    elif train_secs is not None and lead >= train_secs:
        e.append("[training].signal_lead_s must be shorter than [training].time")
    if "{job_id}" not in str(tr.get("pid_dir", "")):
        e.append("[training].pid_dir must contain '{job_id}' so jobs don't share PID files")
    if tr.get("precision") not in ("fp16", "bf16", "fp32"):
        e.append("[training].precision must be fp16, bf16 or fp32")
    elif tr["precision"] == "bf16":
        if any(isinstance(n, Mapping) and n.get("gpu_type") in ("t4", "v100") for n in nodes):
            w.append("[training].precision = bf16, but T4/V100 GPUs have no hardware bf16")
    for k in ("gpus_per_job", "cpus_per_task", "master_port_base"):
        if not isinstance(tr.get(k), int) or tr[k] < 0:
            e.append(f"[training].{k} must be an integer >= 0")
    check_abs("[training].script", tr.get("script"))
    check_abs("[eval].script", cfg["eval"].get("script"))

    # -- slurm ------------------------------------------------------------------------
    s = cfg["slurm"]
    rx = s.get("requeue_exit")
    if not isinstance(rx, int) or not 1 <= rx <= 255:
        e.append("[slurm].requeue_exit must be an integer 1..255")
    elif rx in (1, 2, 126, 127) or rx > 128:
        e.append(
            f"[slurm].requeue_exit {rx} collides with common shell/signal exit codes; "
            "use something like 75"
        )
    if not re.match(r"^\d+-\d+$", str(s.get("srun_port_range", ""))):
        e.append("[slurm].srun_port_range must look like 60001-63000")
    for k in ("state_save_location", "slurmd_spool_dir", "log_dir", "conf_dir"):
        check_abs(f"[slurm].{k}", s.get(k))
    managed = {
        "ClusterName", "SlurmctldHost", "SlurmctldParameters", "AuthType", "CredType",
        "SlurmUser", "SlurmctldPort", "SlurmdPort", "SrunPortRange", "StateSaveLocation",
        "SlurmdSpoolDir", "SlurmctldLogFile", "SlurmdLogFile", "ProctrackType", "TaskPlugin",
        "JobAcctGatherType", "SelectType", "SelectTypeParameters", "SchedulerType",
        "GresTypes", "PriorityType", "AccountingStorageType", "AccountingStorageHost",
        "AccountingStoragePort", "AccountingStorageEnforce", "AccountingStorageTRES",
        "PreemptType", "JobRequeue", "RequeueExit", "SchedulerParameters", "ReturnToService",
        "SlurmctldTimeout", "SlurmdTimeout", "KillWait", "MinJobAge", "Prolog", "PrologFlags",
        "PrologEpilogTimeout", "HealthCheckProgram", "HealthCheckInterval",
        "HealthCheckNodeState", "PriorityDecayHalfLife", "PriorityMaxAge",
    }
    for k in (s.get("extra") or {}):
        if k in managed or k.startswith("PriorityWeight") or k in ("NodeName", "PartitionName"):
            e.append(f"[slurm.extra].{k} is managed by the installer; use the matching TOML key")
    try:
        parse_slurm_time(s.get("priority_decay_half_life"))
        parse_slurm_time(s.get("priority_max_age"))
    except ValueError as ex:
        e.append(f"[slurm] priority time: {ex}")

    # -- health -------------------------------------------------------------------------
    h = cfg["health"]
    if h.get("enabled"):
        check_abs("[health].launch_script", h.get("launch_script"))
        if not (h.get("prolog") or h.get("periodic")):
            w.append("[health].enabled = true but prolog and periodic are both false")
    if h.get("node_state") not in ("ANY", "IDLE", "ALLOC", "MIXED", "NONDRAINED_IDLE"):
        e.append("[health].node_state must be ANY, IDLE, ALLOC, MIXED or NONDRAINED_IDLE")

    # -- secrets ------------------------------------------------------------------------
    sec = cfg["secrets"]
    if sec.get("munge_key_source") not in ("file", "ssm"):
        e.append("[secrets].munge_key_source must be 'file' or 'ssm'")
    elif sec["munge_key_source"] == "ssm":
        if not sec.get("ssm_parameter"):
            e.append("[secrets].ssm_parameter is required when munge_key_source = 'ssm'")
        if not sec.get("ssm_region"):
            e.append("[secrets].ssm_region is required when munge_key_source = 'ssm'")
    check_abs("[secrets].munge_key_file", sec.get("munge_key_file"))

    # -- storage ------------------------------------------------------------------------
    sc = cfg["storage"]["scratch"]
    if sc.get("type") not in ("nvme", "dir"):
        e.append("[storage.scratch].type must be 'nvme' or 'dir'")
    if sc.get("filesystem") not in ("xfs", "ext4"):
        e.append("[storage.scratch].filesystem must be 'xfs' or 'ext4'")
    check_abs("[storage.scratch].mount", sc.get("mount"))
    sh = cfg["storage"]["shared"]
    if sh.get("type") not in ("none", "efs", "nfs"):
        e.append("[storage.shared].type must be 'none', 'efs' or 'nfs'")
    elif sh["type"] == "efs":
        if not sh.get("efs_id"):
            e.append("[storage.shared].efs_id is required when type = 'efs'")
        if not sh.get("region"):
            e.append("[storage.shared].region is required when type = 'efs'")
    elif sh["type"] == "nfs" and not sh.get("nfs_server"):
        e.append("[storage.shared].nfs_server is required when type = 'nfs'")
    if sh.get("type") != "none":
        check_abs("[storage.shared].mount", sh.get("mount"))

    # -- checkpoints / logs -------------------------------------------------------------
    ck = cfg["checkpoints"]
    check_abs("[checkpoints].local_dir", ck.get("local_dir"))
    if ck.get("durable") not in ("s3", "shared", "none"):
        e.append("[checkpoints].durable must be 's3', 'shared' or 'none'")
    elif ck["durable"] == "s3":
        if not str(ck.get("s3_uri", "")).startswith("s3://"):
            e.append("[checkpoints].s3_uri must start with s3:// when durable = 's3'")
    elif ck["durable"] == "shared":
        if sh.get("type") == "none":
            e.append("[checkpoints].durable = 'shared' needs [storage.shared].type efs or nfs")
        sd = ck.get("shared_dir", "")
        if not sd or not str(sd).startswith(str(sh.get("mount", "")).rstrip("/") + "/"):
            e.append("[checkpoints].shared_dir must be a path under [storage.shared].mount")
    else:
        w.append(
            "[checkpoints].durable = 'none': a node failure loses all checkpoints, "
            "and eval cannot read training checkpoints"
        )
    for path_key, value in (("checkpoints.local_dir", ck.get("local_dir")), ("logs.dir", cfg["logs"].get("dir"))):
        if isinstance(value, str) and sh.get("type") == "none" and value.startswith(str(sh.get("mount")) + "/"):
            e.append(f"[{path_key}] is under {sh.get('mount')} but [storage.shared].type = 'none'")
    check_abs("[logs].dir", cfg["logs"].get("dir"))

    # -- code / python ------------------------------------------------------------------
    co = cfg["code"]
    if co.get("source") not in ("none", "local", "git"):
        e.append("[code].source must be 'none', 'local' or 'git'")
    elif co["source"] == "git" and not co.get("git_url"):
        e.append("[code].git_url is required when source = 'git'")
    check_abs("[code].dest", co.get("dest"))
    py = cfg["python"]
    for k in ("venv", "python_install_dir", "cache_dir", "cuda_home", "torch_extensions_dir"):
        check_abs(f"[python].{k}", py.get(k))
    if not re.match(r"^\d+\.\d+\.\d+$", str(py.get("uv_version", ""))):
        e.append("[python].uv_version must be an exact version like 0.13.6")

    # -- tests ---------------------------------------------------------------------------
    t = cfg["tests"]
    if t.get("limits_partition") and t["limits_partition"] not in part_by_name:
        w.append(f"[tests].limits_partition {t['limits_partition']!r} is not defined; T07/T08 will be skipped")
    if t.get("limits_qos") and t["limits_qos"] not in qos_by_name:
        w.append(f"[tests].limits_qos {t['limits_qos']!r} is not defined; T07/T08 will be skipped")

    return e, w


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


class Config(Mapping):
    """Read-only view of a normalized, validated cluster config."""

    def __init__(self, data: dict[str, Any], path: Path | None = None, warnings_: list[str] | None = None):
        self._data = data
        self.path = path
        self.warnings = warnings_ or []

    # Mapping protocol
    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def get(self, key, default=None):
        """Dotted lookup: cfg.get("slurm.requeue_exit")."""
        cur: Any = self._data
        for part in str(key).split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    # -- lookups -----------------------------------------------------------------
    def node(self, name: str) -> dict[str, Any]:
        for n in self._data["nodes"]:
            if n["name"] == name:
                return n
        raise KeyError(f"no node named {name!r} in {self.path}")

    def nodes_by_role(self, role: str) -> list[dict[str, Any]]:
        return [n for n in self._data["nodes"] if n["role"] == role]

    def this_node(self) -> dict[str, Any] | None:
        short = socket.gethostname().split(".")[0]
        return next((n for n in self._data["nodes"] if n["name"] == short), None)

    def partition(self, name: str) -> dict[str, Any]:
        for p in self._data["partitions"]:
            if p["name"] == name:
                return p
        raise KeyError(f"no partition named {name!r}")

    def account(self, name: str) -> dict[str, Any]:
        for a in self._data["accounts"]:
            if a["name"] == name:
                return a
        raise KeyError(f"no account named {name!r}")

    def qos(self, name: str) -> dict[str, Any]:
        for q in self._data["qos"]:
            if q["name"] == name:
                return q
        raise KeyError(f"no QOS named {name!r}")

    def users_in(self, account: str) -> list[dict[str, Any]]:
        return [u for u in self._data["users"] if u["account"] == account]

    # -- values training code needs ------------------------------------------------
    @property
    def requeue_exit(self) -> int:
        return int(self._data["slurm"]["requeue_exit"])

    @property
    def signal_lead_s(self) -> int:
        return int(self._data["training"]["signal_lead_s"])

    def pid_dir(self, job_id: str | int | None = None) -> Path:
        jid = job_id if job_id is not None else os.environ.get("SLURM_JOB_ID", "nojob")
        return Path(self._data["training"]["pid_dir"].format(job_id=jid))

    @property
    def checkpoints(self) -> dict[str, Any]:
        return copy.deepcopy(self._data["checkpoints"])


def _resolve_path(path: str | os.PathLike | None) -> Path:
    if path:
        return Path(path)
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    return INSTALLED_PATH


def load(path: str | os.PathLike | None = None, *, strict: bool = True) -> Config:
    """Load, fill defaults and validate.

    Path order: argument, $TEAM7_CONFIG, /etc/team7/cluster.toml.
    Raises ConfigError on validation errors when strict is True.
    Warns (warnings.warn) if the file differs from the installed copy.
    """
    p = _resolve_path(path)
    try:
        raw = load_raw(p)
    except FileNotFoundError:
        raise ConfigError([f"file not found: {p}"], p) from None
    except tomllib.TOMLDecodeError as ex:
        raise ConfigError([f"TOML syntax: {ex}"], p) from None
    cfg = normalize(raw)
    errors, warns = validate(cfg)
    if errors and strict:
        raise ConfigError(errors, p)
    try:
        if INSTALLED_PATH.exists() and p.resolve() != INSTALLED_PATH.resolve():
            if file_sha256(p) != file_sha256(INSTALLED_PATH):
                warnings.warn(
                    f"{p} differs from the installed {INSTALLED_PATH}; "
                    "Slurm may be configured with other values (run the installer)",
                    stacklevel=2,
                )
    except OSError:
        pass
    return Config(cfg, p, warns)
