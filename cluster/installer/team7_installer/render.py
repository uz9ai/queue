"""Turn the cluster TOML into config files and scripts.

Every function takes the normalized config (team7config.Config) and returns
file content. Nothing here touches the system, so `install.sh render` can
show exactly what would be installed.
"""

from __future__ import annotations

import shlex
from pathlib import Path


ROOT = Path("/opt/team7")
BIN_DIR = ROOT / "bin"
SLURM_SCRIPTS = ROOT / "slurm"
JOBS_DIR = ROOT / "jobs"
CLUSTER_COPY = ROOT / "cluster"
ENV_FILE = Path("/etc/team7/team7.env")
SCRATCH_SCRIPT = BIN_DIR / "team7-scratch-setup"
PROLOG = SLURM_SCRIPTS / "prolog.sh"
HEALTHCHECK = SLURM_SCRIPTS / "healthcheck.sh"

GENERATED = "Generated from cluster.toml by the Team 7 installer. Edit the TOML, not this file."


def _yn(v: bool) -> str:
    return "yes" if v else "no"


# ---------------------------------------------------------------------------
# Slurm
# ---------------------------------------------------------------------------


def node_line(n: dict) -> str:
    parts = [f"NodeName={n['name']}", f"NodeAddr={n['ip']}", f"CPUs={n['cpus']}"]
    if n.get("sockets") is not None:
        parts += [
            f"Sockets={n['sockets']}",
            f"CoresPerSocket={n['cores_per_socket']}",
            f"ThreadsPerCore={n['threads_per_core']}",
        ]
    parts.append(f"RealMemory={n['real_memory_mb']}")
    if n["gpus"] > 0:
        parts.append(f"Gres=gpu:{n['gpu_type']}:{n['gpus']}")
    features = [n["role"]] + ([n["gpu_type"]] if n.get("gpu_type") else []) + list(n.get("features") or [])
    parts.append("Feature=" + ",".join(dict.fromkeys(features)))
    parts.append(f"Weight={n['weight']}")
    parts.append("State=UNKNOWN")
    return " ".join(parts)


def partition_line(p: dict) -> str:
    parts = [f"PartitionName={p['name']}", "Nodes=" + ",".join(p["nodes"])]
    if p.get("allow_accounts"):
        parts.append("AllowAccounts=" + ",".join(p["allow_accounts"]))
    if p.get("allow_qos"):
        parts.append("AllowQos=" + ",".join(p["allow_qos"]))
    parts.append(f"MaxTime={p['max_time']}")
    if p.get("default_time"):
        parts.append(f"DefaultTime={p['default_time']}")
    parts.append("Default=" + ("YES" if p.get("default") else "NO"))
    for k, v in (p.get("extra") or {}).items():
        parts.append(f"{k}={v}")
    parts.append(f"State={p['state']}")
    return " ".join(parts)


def slurm_conf(cfg) -> str:
    s = cfg["slurm"]
    c = cfg["controller"]
    h = cfg["health"]
    w = s["priority_weights"]
    kv: list[tuple[str, object]] = [
        ("ClusterName", cfg["cluster"]["name"]),
        ("SlurmctldHost", f"{c['name']}({c['ip']})"),
        ("SlurmctldParameters", "enable_configless"),
        ("AuthType", "auth/munge"),
        ("CredType", "cred/munge"),
        ("SlurmUser", "slurm"),
        ("SlurmdUser", "root"),
        ("SlurmctldPort", s["ctld_port"]),
        ("SlurmdPort", s["slurmd_port"]),
        ("SrunPortRange", s["srun_port_range"]),
        ("StateSaveLocation", s["state_save_location"]),
        ("SlurmdSpoolDir", s["slurmd_spool_dir"]),
        ("SlurmctldLogFile", f"{s['log_dir']}/slurmctld.log"),
        ("SlurmdLogFile", f"{s['log_dir']}/slurmd.log"),
        ("", "# Resources and isolation"),
        ("ProctrackType", "proctrack/cgroup"),
        ("TaskPlugin", s["task_plugin"]),
        ("JobAcctGatherType", "jobacct_gather/cgroup"),
        ("JobAcctGatherFrequency", "task=30"),
        ("SelectType", "select/cons_tres"),
        ("SelectTypeParameters", s["select_type_parameters"]),
        ("GresTypes", "gpu"),
        ("", "# Scheduling and priority"),
        ("SchedulerType", s["scheduler_type"]),
    ]
    if s.get("scheduler_parameters"):
        kv.append(("SchedulerParameters", ",".join(s["scheduler_parameters"])))
    kv += [
        ("PriorityType", "priority/multifactor"),
        ("PriorityDecayHalfLife", s["priority_decay_half_life"]),
        ("PriorityMaxAge", s["priority_max_age"]),
        ("PriorityWeightFairshare", w["fairshare"]),
        ("PriorityWeightAge", w["age"]),
        ("PriorityWeightQOS", w["qos"]),
        ("PriorityWeightPartition", w["partition"]),
        ("PriorityWeightJobSize", w["job_size"]),
        ("PreemptType", s["preempt_type"]),
        ("", "# Accounting and limits"),
        ("AccountingStorageType", "accounting_storage/slurmdbd"),
        ("AccountingStorageHost", c["name"]),
        ("AccountingStoragePort", s["dbd_port"]),
        ("AccountingStorageEnforce", ",".join(s["accounting_enforce"])),
    ]
    if s.get("accounting_tres"):
        kv.append(("AccountingStorageTRES", ",".join(s["accounting_tres"])))
    kv += [
        ("JobCompType", "jobcomp/none"),
        ("", "# Requeue and failure handling"),
        ("JobRequeue", 1 if s["job_requeue"] else 0),
        ("RequeueExit", s["requeue_exit"]),
        ("ReturnToService", s["return_to_service"]),
        ("SlurmctldTimeout", s["slurmctld_timeout"]),
        ("SlurmdTimeout", s["slurmd_timeout"]),
        ("KillWait", s["kill_wait"]),
        ("MinJobAge", s["min_job_age"]),
    ]
    if h["enabled"]:
        kv.append(("", "# Health checks"))
        if h["prolog"]:
            kv += [
                ("Prolog", str(PROLOG)),
                ("PrologFlags", "Alloc"),
                ("PrologEpilogTimeout", h["prolog_timeout_s"]),
            ]
        if h["periodic"]:
            kv += [
                ("HealthCheckProgram", str(HEALTHCHECK)),
                ("HealthCheckInterval", h["interval_s"]),
                ("HealthCheckNodeState", h["node_state"]),
            ]
    if s.get("extra"):
        kv.append(("", "# [slurm.extra]"))
        kv += list(s["extra"].items())

    lines = [f"# {GENERATED}", ""]
    for k, v in kv:
        if not k:
            lines += ["", str(v)]
        else:
            lines.append(f"{k}={v}")
    lines += ["", "# Nodes"]
    lines += [node_line(n) for n in cfg["nodes"]]
    lines += ["", "# Partitions"]
    lines += [partition_line(p) for p in cfg["partitions"]]
    return "\n".join(lines) + "\n"


def gres_conf(cfg) -> str:
    lines = [f"# {GENERATED}"]
    for n in cfg["nodes"]:
        if n["gpus"] <= 0:
            continue
        files = "/dev/nvidia0" if n["gpus"] == 1 else f"/dev/nvidia[0-{n['gpus'] - 1}]"
        lines.append(f"NodeName={n['name']} Name=gpu Type={n['gpu_type']} File={files}")
    return "\n".join(lines) + "\n"


def cgroup_conf(cfg) -> str:
    cg = cfg["slurm"]["cgroup"]
    return (
        f"# {GENERATED}\n"
        "CgroupPlugin=autodetect\n"
        f"ConstrainCores={_yn(cg['constrain_cores'])}\n"
        f"ConstrainRAMSpace={_yn(cg['constrain_ram_space'])}\n"
        f"ConstrainDevices={_yn(cg['constrain_devices'])}\n"
    )


def slurmdbd_conf(cfg, db_password: str) -> str:
    s = cfg["slurm"]
    d = cfg["database"]
    return (
        f"# {GENERATED}\n"
        "AuthType=auth/munge\n"
        f"DbdHost={cfg['controller']['name']}\n"
        f"DbdPort={s['dbd_port']}\n"
        "SlurmUser=slurm\n"
        f"LogFile={s['log_dir']}/slurmdbd.log\n"
        "StorageType=accounting_storage/mysql\n"
        "StorageHost=localhost\n"
        "StoragePort=3306\n"
        f"StorageUser={d['user']}\n"
        f"StoragePass={db_password}\n"
        f"StorageLoc={d['name']}\n"
    )


def mariadb_cnf(cfg) -> str:
    d = cfg["database"]
    return (
        f"# {GENERATED}\n"
        "[mysqld]\n"
        "bind-address=127.0.0.1\n"
        f"innodb_buffer_pool_size={d['innodb_buffer_pool_size']}\n"
        f"innodb_lock_wait_timeout={d['innodb_lock_wait_timeout']}\n"
    )


# ---------------------------------------------------------------------------
# Host files
# ---------------------------------------------------------------------------

HOSTS_BEGIN = "# BEGIN team7 (managed by the Team 7 installer)"
HOSTS_END = "# END team7"


def hosts_block(cfg) -> str:
    lines = [HOSTS_BEGIN, f"{cfg['controller']['ip']} {cfg['controller']['name']}"]
    lines += [f"{n['ip']} {n['name']}" for n in cfg["nodes"]]
    lines.append(HOSTS_END)
    return "\n".join(lines)


def merge_hosts(existing: str, block: str) -> str:
    out, skip = [], False
    for line in existing.splitlines():
        if line.strip() == HOSTS_BEGIN:
            skip = True
            continue
        if skip and line.strip() == HOSTS_END:
            skip = False
            continue
        if not skip:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out + ["", block]) + "\n"


def cloud_cfg() -> str:
    return f"# {GENERATED}\npreserve_hostname: true\nmanage_etc_hosts: false\n"


def chrony_sources(cfg) -> str:
    return f"# {GENERATED}\nserver {cfg['cluster']['ntp_server']} prefer iburst minpoll 4 maxpoll 4\n"


def team7_env(cfg) -> str:
    py = cfg["python"]
    return (
        f"# {GENERATED}\n"
        "TEAM7_CONFIG=/etc/team7/cluster.toml\n"
        f"TEAM7_ROOT={ROOT}\n"
        f"TEAM7_VENV={py['venv']}\n"
        f"CUDA_HOME={py['cuda_home']}\n"
        f"TORCH_EXTENSIONS_DIR={py['torch_extensions_dir']}\n"
        f"TORCH_CUDA_ARCH_LIST={py['cuda_arch_list']}\n"
    )


def profile_sh() -> str:
    return (
        f"# {GENERATED}\n"
        "if [ -r /etc/team7/team7.env ]; then\n"
        "  set -a\n"
        "  . /etc/team7/team7.env\n"
        "  set +a\n"
        "fi\n"
    )


def team7_config_wrapper() -> str:
    return (
        "#!/bin/sh\n"
        f"# {GENERATED}\n"
        f'PYTHONPATH="{CLUSTER_COPY}/team7config${{PYTHONPATH:+:$PYTHONPATH}}" '
        'exec /usr/bin/python3 -m team7config "$@"\n'
    )


# ---------------------------------------------------------------------------
# GPU node files
# ---------------------------------------------------------------------------


def scratch_setup_sh(cfg) -> str:
    sc = cfg["storage"]["scratch"]
    group = cfg["unix"]["group"]
    ckpt = cfg["checkpoints"]["local_dir"]
    logs = cfg["logs"]["dir"]
    mkfs = "mkfs.xfs -f" if sc["filesystem"] == "xfs" else "mkfs.ext4 -F -q"
    return f"""#!/bin/bash
# {GENERATED}
# Prepares the scratch directories used for checkpoints and logs.
#
#   type = "dir"  : $MOUNT is expected to be an already-mounted filesystem
#                   (e.g. the Deep Learning AMI's NVMe at /opt/dlami/nvme).
#                   This script only creates the job directories under it.
#   type = "nvme" : format a BLANK instance-store device and mount it at $MOUNT.
#                   Refuses to touch a device that is already mounted or that
#                   already holds a filesystem/LVM signature.
# Instance-store NVMe is erased on stop/start, so this runs on every boot.
set -euo pipefail

TYPE={shlex.quote(sc['type'])}
MOUNT={shlex.quote(sc['mount'])}
DEVICE={shlex.quote(sc['device'])}

if [ "$TYPE" = "dir" ]; then
  if ! mountpoint -q "$MOUNT" && [ "$MOUNT" != "${{MOUNT%/*}}" ]; then
    parent=$(df -P "$MOUNT" 2>/dev/null | awk 'NR==2 {{print $6}}' || true)
    echo "team7-scratch: type=dir, using existing filesystem at $MOUNT (on ${{parent:-/}})"
  fi
  mkdir -p "$MOUNT"
elif mountpoint -q "$MOUNT"; then
  echo "team7-scratch: $MOUNT already mounted"
else
  if [ "$DEVICE" = "auto" ]; then
    # Only pick an instance-store device that is NOT already in use: no mount,
    # no partitions, and no existing filesystem/LVM signature.
    DEVICE=""
    while read -r dev model; do
      [ -b "$dev" ] || continue
      case "$model" in *"Instance Storage"*) ;; *) continue ;; esac
      findmnt -S "$dev" >/dev/null 2>&1 && continue
      lsblk -no NAME "$dev" | tail -n +2 | grep -q . && continue   # has partitions/LVM children
      if blkid "$dev" >/dev/null 2>&1; then continue; fi           # already has a signature
      DEVICE="$dev"; break
    done < <(lsblk -dpno NAME,MODEL)
  fi
  if [ -z "$DEVICE" ] || [ ! -b "$DEVICE" ]; then
    echo "team7-scratch: no blank instance-store NVMe device found." >&2
    echo "  If this AMI already mounts its NVMe (e.g. the Deep Learning AMI at" >&2
    echo "  /opt/dlami/nvme), set [storage.scratch].type = \\"dir\\" and point mount/" >&2
    echo "  checkpoints/logs at that path instead of reformatting the disk." >&2
    exit 1
  fi
  # Guard: never reformat a device that is mounted or already has a signature.
  if findmnt -S "$DEVICE" >/dev/null 2>&1; then
    echo "team7-scratch: refusing to reformat $DEVICE; it is already mounted (use type = \\"dir\\")" >&2
    exit 1
  fi
  if blkid "$DEVICE" >/dev/null 2>&1; then
    sig=$(blkid -o value -s TYPE "$DEVICE" 2>/dev/null || echo unknown)
    echo "team7-scratch: refusing to reformat $DEVICE; it already holds a '$sig' signature." >&2
    echo "  Point [storage.scratch] at the existing mount with type = \\"dir\\", or set" >&2
    echo "  [storage.scratch].device explicitly to a known-blank device to overwrite it." >&2
    exit 1
  fi
  echo "team7-scratch: formatting blank device $DEVICE"
  {mkfs} "$DEVICE"
  mkdir -p "$MOUNT"
  mount -o noatime "$DEVICE" "$MOUNT"
  echo "team7-scratch: mounted $DEVICE on $MOUNT"
fi

if ! mountpoint -q "$MOUNT" && [ "$TYPE" = "nvme" ]; then
  echo "team7-scratch: $MOUNT is not a mount point after setup" >&2
  exit 1
fi

mkdir -p {shlex.quote(ckpt)}
chown root:{shlex.quote(group)} {shlex.quote(ckpt)}
chmod 2775 {shlex.quote(ckpt)}
mkdir -p {shlex.quote(logs)}
chmod 1777 {shlex.quote(logs)}
"""


def scratch_unit() -> str:
    return f"""# {GENERATED}
[Unit]
Description=Team 7 scratch disk (format and mount instance store)
After=local-fs.target
Before=slurmd.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={SCRATCH_SCRIPT}

[Install]
WantedBy=multi-user.target
"""


def slurmd_default(cfg) -> str:
    c = cfg["controller"]
    return f'# {GENERATED}\nSLURMD_OPTIONS="--conf-server {c["name"]}:{cfg["slurm"]["ctld_port"]}"\n'


def slurmd_dropin() -> str:
    return f"""# {GENERATED}
[Unit]
# Configless: there is no local slurm.conf, so drop any condition on it.
ConditionPathExists=
Wants=network-online.target
After=network-online.target munge.service team7-scratch.service
Requires=team7-scratch.service

[Service]
EnvironmentFile=-/etc/default/slurmd
"""


def prolog_sh(cfg) -> str:
    h = cfg["health"]
    return f"""#!/bin/bash
# {GENERATED}
# Runs as root on the node before each job (PrologFlags=Alloc).
# A non-zero exit makes Slurm drain the node and requeue the job.
set -u
LOG_DIR={shlex.quote(h['log_dir'])}
mkdir -p "$LOG_DIR"
timeout {int(h['check_timeout_s'])} {shlex.quote(h['launch_script'])} {h['quick_args']} \\
  >>"$LOG_DIR/prolog.jsonl" 2>>"$LOG_DIR/prolog.log"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "$(date -u +%FT%TZ) job=${{SLURM_JOB_ID:-}} health check failed rc=$rc" >>"$LOG_DIR/prolog.log"
fi
exit "$rc"
"""


def healthcheck_sh(cfg) -> str:
    h = cfg["health"]
    return f"""#!/bin/bash
# {GENERATED}
# Slurm HealthCheckProgram. Slurm ignores this script's exit code, so the
# script drains the node itself when the check fails.
set -u
LOG_DIR={shlex.quote(h['log_dir'])}
NODE=$(hostname -s)
mkdir -p "$LOG_DIR"
timeout {int(h['check_timeout_s'])} {shlex.quote(h['launch_script'])} {h['quick_args']} \\
  >>"$LOG_DIR/healthcheck.jsonl" 2>>"$LOG_DIR/healthcheck.log"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "$(date -u +%FT%TZ) health check failed rc=$rc, draining $NODE" >>"$LOG_DIR/healthcheck.log"
  scontrol update nodename="$NODE" state=drain reason="health: quick check failed rc=$rc (see $LOG_DIR/healthcheck.log)"
fi
exit 0
"""


# ---------------------------------------------------------------------------
# Job templates
# ---------------------------------------------------------------------------


def _env_block(cfg) -> str:
    py = cfg["python"]
    return f"""export TEAM7_CONFIG=/etc/team7/cluster.toml
export CUDA_HOME={shlex.quote(py['cuda_home'])}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR={shlex.quote(py['torch_extensions_dir'])}
export TORCH_CUDA_ARCH_LIST={shlex.quote(py['cuda_arch_list'])}"""


def _chdir(cfg) -> str:
    return cfg["code"]["dest"] if cfg["code"]["source"] != "none" else cfg["logs"]["dir"]


def _template_node(cfg, partition: str) -> dict | None:
    for p in cfg["partitions"]:
        if p["name"] == partition and p["nodes"]:
            for n in cfg["nodes"]:
                if n["name"] == p["nodes"][0]:
                    return n
    return None


def train_sbatch(cfg) -> str:
    t = cfg["training"]
    node = _template_node(cfg, t["partition"]) or {"gpus": 1, "cpus": 1}
    gpus = t["gpus_per_job"] or node["gpus"]
    cpus = t["cpus_per_task"] or node["cpus"]
    pid_dir = t["pid_dir"].replace("{job_id}", "${SLURM_JOB_ID}")
    args = " ".join(shlex.quote(str(a)) for a in t["args"])
    venv = cfg["python"]["venv"]
    rx = cfg["slurm"]["requeue_exit"]
    gres = f"#SBATCH --gres=gpu:{gpus}\n" if gpus else ""
    return f"""#!/bin/bash
# {GENERATED}
#
# Training job: DeepSpeed launcher in the batch step, checkpoint on SIGUSR1,
# self-requeue with exit code {rx} (see queue_design.md section 8).
#
#   sbatch {JOBS_DIR}/train.sbatch [extra args for the training script]
#   TEAM7_TRAIN_SCRIPT=/path/other.py sbatch --export=ALL {JOBS_DIR}/train.sbatch
#
#SBATCH --job-name={t['job_name']}
#SBATCH --account={t['account']}
#SBATCH --partition={t['partition']}
#SBATCH --qos={t['qos']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
{gres}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={t['mem']}
#SBATCH --time={t['time']}
#SBATCH --requeue
#SBATCH --signal=B:USR1@{t['signal_lead_s']}
#SBATCH --open-mode=append
#SBATCH --output={cfg['logs']['dir']}/%x-%j.out
#SBATCH --chdir={_chdir(cfg)}

set -u
REQUEUE_EXIT={rx}
{_env_block(cfg)}
export TEAM7_PID_DIR="{pid_dir}"
TRAIN_SCRIPT="${{TEAM7_TRAIN_SCRIPT:-{t['script']}}}"
DS_CONFIG="${{TEAM7_DS_CONFIG-{t['deepspeed_config']}}}"   # set TEAM7_DS_CONFIG= (empty) to pass none

# A requeued job keeps its job ID, so clear PID files left by the last segment.
mkdir -p "$TEAM7_PID_DIR"
rm -f "$TEAM7_PID_DIR"/rank-*.pid
trap 'rm -rf "$TEAM7_PID_DIR"' EXIT

# Forward SIGUSR1 only to the training ranks. The DeepSpeed launcher processes
# do not handle SIGUSR1 and would die if they received it.
forward_usr1() {{
  echo "$(date -u +%FT%TZ) SIGUSR1: forwarding to training ranks" >&2
  local f
  for f in "$TEAM7_PID_DIR"/rank-*.pid; do
    [ -e "$f" ] && kill -USR1 "$(cat "$f")" 2>/dev/null
  done
}}
trap forward_usr1 USR1

echo "job=$SLURM_JOB_ID restarts=${{SLURM_RESTART_COUNT:-0}} node=$(hostname -s) start=$(date -u +%FT%TZ)"

script_args=({args})
if [ -n "$DS_CONFIG" ]; then
  script_args=(--deepspeed_config "$DS_CONFIG" "${{script_args[@]}}")
fi

# No --num_gpus: the launcher then uses the GPUs in CUDA_VISIBLE_DEVICES (set by Slurm).
{venv}/bin/deepspeed --master_port=$(( {t['master_port_base']} + SLURM_JOB_ID % 20000 )) \\
  "$TRAIN_SCRIPT" "${{script_args[@]}}" "$@" &
pid=$!

# `wait` returns early when the USR1 trap runs. Keep waiting until the launcher exits.
while true; do
  wait "$pid"
  rc=$?
  kill -0 "$pid" 2>/dev/null || break
done

echo "exit=$rc end=$(date -u +%FT%TZ)"
if [ "$rc" -eq "$REQUEUE_EXIT" ]; then
  echo "checkpoint saved; Slurm requeues this job (RequeueExit=$REQUEUE_EXIT)"
fi
exit "$rc"
"""


def eval_sbatch(cfg) -> str:
    e = cfg["eval"]
    args = " ".join(shlex.quote(str(a)) for a in e["args"])
    gres = f"#SBATCH --gres=gpu:{e['gpus_per_job']}\n" if e["gpus_per_job"] else ""
    cpus = f"#SBATCH --cpus-per-task={e['cpus_per_task']}\n" if e["cpus_per_task"] else ""
    return f"""#!/bin/bash
# {GENERATED}
#
# Evaluation job. Reads only checkpoints with a COMPLETE marker.
#
#   sbatch {JOBS_DIR}/eval.sbatch [extra args for the eval script]
#
#SBATCH --job-name={e['job_name']}
#SBATCH --account={e['account']}
#SBATCH --partition={e['partition']}
#SBATCH --qos={e['qos']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
{gres}{cpus}#SBATCH --mem={e['mem']}
#SBATCH --time={e['time']}
#SBATCH --output={cfg['logs']['dir']}/%x-%j.out
#SBATCH --chdir={_chdir(cfg)}

set -u
{_env_block(cfg)}
EVAL_SCRIPT="${{TEAM7_EVAL_SCRIPT:-{e['script']}}}"
script_args=({args})

echo "job=$SLURM_JOB_ID node=$(hostname -s) start=$(date -u +%FT%TZ)"
exec {cfg['python']['venv']}/bin/python "$EVAL_SCRIPT" "${{script_args[@]}}" "$@"
"""


def debug_sbatch(cfg) -> str:
    d = cfg["debug"]
    gres = f"#SBATCH --gres=gpu:{d['gpus_per_job']}\n" if d["gpus_per_job"] else ""
    cpus = f"#SBATCH --cpus-per-task={d['cpus_per_task']}\n" if d["cpus_per_task"] else ""
    return f"""#!/bin/bash
# {GENERATED}
#
# Short debug job on your team's own node. Pass the partition:
#   sbatch -p train {JOBS_DIR}/debug.sbatch            # GPU and environment check
#   sbatch -p eval  {JOBS_DIR}/debug.sbatch python my_test.py
#
#SBATCH --job-name=debug
#SBATCH --qos={d['qos']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
{gres}{cpus}#SBATCH --mem={d['mem']}
#SBATCH --time={d['time']}
#SBATCH --output={cfg['logs']['dir']}/%x-%j.out
#SBATCH --chdir={_chdir(cfg)}

set -u
{_env_block(cfg)}
export PATH="{cfg['python']['venv']}/bin:$PATH"

echo "job=$SLURM_JOB_ID user=$(id -un) node=$(hostname -s) gpus=${{CUDA_VISIBLE_DEVICES:-none}}"
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
nvidia-smi -L
python -c "import torch, deepspeed; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('deepspeed', deepspeed.__version__)"
"""


# ---------------------------------------------------------------------------
# Everything render can show
# ---------------------------------------------------------------------------


def all_files(cfg, db_password: str = "<generated on the controller>") -> dict[str, str]:
    files = {
        "etc/slurm/slurm.conf": slurm_conf(cfg),
        "etc/slurm/gres.conf": gres_conf(cfg),
        "etc/slurm/cgroup.conf": cgroup_conf(cfg),
        "etc/slurm/slurmdbd.conf": slurmdbd_conf(cfg, db_password),
        "etc/mysql/mariadb.conf.d/60-team7-slurm.cnf": mariadb_cnf(cfg),
        "etc/hosts.team7-block": hosts_block(cfg) + "\n",
        "etc/team7/team7.env": team7_env(cfg),
        "etc/default/slurmd": slurmd_default(cfg),
        "etc/systemd/system/slurmd.service.d/team7.conf": slurmd_dropin(),
        "etc/systemd/system/team7-scratch.service": scratch_unit(),
        "opt/team7/bin/team7-scratch-setup": scratch_setup_sh(cfg),
        "opt/team7/jobs/train.sbatch": train_sbatch(cfg),
        "opt/team7/jobs/eval.sbatch": eval_sbatch(cfg),
        "opt/team7/jobs/debug.sbatch": debug_sbatch(cfg),
    }
    if cfg["health"]["enabled"]:
        files["opt/team7/slurm/prolog.sh"] = prolog_sh(cfg)
        files["opt/team7/slurm/healthcheck.sh"] = healthcheck_sh(cfg)
    return files
