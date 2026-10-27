"""Steps that run on every host: preflight, hostname, users, packages, time,
shared storage, code, and the installed copy of the TOML."""

from __future__ import annotations

import grp
import os
import pwd
import re
import shutil
from pathlib import Path

from team7config import file_sha256

from .. import render
from ..util import (
    ETC,
    INSTALLED_TOML,
    Ctx,
    InstallError,
    apt_install,
    ensure_dir,
    header,
    info,
    ok,
    out,
    restart,
    run,
    systemctl,
    warn,
    write_file,
)

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _os_release() -> dict[str, str]:
    data = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k] = v.strip().strip('"')
    except OSError:
        pass
    return data


def _local_ipv4s(ctx: Ctx) -> set[str]:
    text = out(ctx, ["ip", "-4", "-o", "addr", "show"])
    return set(re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", text))


def preflight(ctx: Ctx) -> None:
    header("Preflight")
    cfg = ctx.cfg
    if os.geteuid() != 0:
        ctx.fail_or_warn("run as root: sudo ./install.sh ...")
    for w in cfg.warnings:
        warn(w)

    osr = _os_release()
    want = cfg["cluster"]["os_version"]
    if osr.get("ID") != "ubuntu" or osr.get("VERSION_ID") != want:
        ctx.fail_or_warn(
            f"this host runs {osr.get('PRETTY_NAME', 'an unknown OS')}; "
            f"[cluster].os_version expects Ubuntu {want}"
        )
    else:
        ok(f"Ubuntu {want}")

    expected_ip = cfg["controller"]["ip"] if ctx.is_controller else ctx.node["ip"]
    ips = _local_ipv4s(ctx)
    if expected_ip not in ips:
        ctx.fail_or_warn(
            f"this host has IPs {sorted(ips) or 'unknown'}, but cluster.toml says "
            f"{ctx.hostname} is {expected_ip}. Wrong host, or wrong role/name argument?"
        )
    else:
        ok(f"{ctx.hostname} has IP {expected_ip}")

    if ctx.is_node:
        _check_gpus(ctx)


def _check_gpus(ctx: Ctx) -> None:
    n = ctx.node
    if n["gpus"] == 0:
        ok("node has gpus = 0, skipping GPU check")
        return
    if shutil.which("nvidia-smi") is None:
        ctx.fail_or_warn("nvidia-smi not found; use the Ubuntu 24.04 GPU AMI (NVIDIA driver installed)")
        return
    listing = out(ctx, ["nvidia-smi", "-L"])
    gpus = [line for line in listing.splitlines() if line.startswith("GPU ")]
    if len(gpus) != n["gpus"]:
        ctx.fail_or_warn(f"nvidia-smi shows {len(gpus)} GPU(s), cluster.toml says {n['gpus']}")
        return
    match = (n.get("gpu_name_match") or n["gpu_type"]).lower()
    bad = [g for g in gpus if match not in g.lower()]
    if bad:
        ctx.fail_or_warn(f"GPU name does not contain {match!r}: {bad[0]} (set gpu_name_match)")
        return
    ok(f"{len(gpus)} GPU(s): {gpus[0].split('(')[0].strip()}")


def hardware(ctx: Ctx) -> None:
    """Compare cluster.toml with `slurmd -C`. Runs after packages are installed."""
    header("Hardware vs cluster.toml")
    n = ctx.node
    line = out(ctx, ["slurmd", "-C"]).splitlines()
    line = line[0] if line else ""
    if not line:
        ctx.fail_or_warn("`slurmd -C` gave no output (is slurmd installed?)")
        return
    found = dict(re.findall(r"(\w+)=(\S+)", line))
    info(f"slurmd -C: {line}")
    problems = []
    checks = [("cpus", "CPUs"), ("sockets", "SocketsPerBoard"), ("cores_per_socket", "CoresPerSocket"),
              ("threads_per_core", "ThreadsPerCore")]
    for key, slurm_key in checks:
        if n.get(key) is not None and slurm_key in found and int(found[slurm_key]) != int(n[key]):
            problems.append(f"{key} = {n[key]} but slurmd reports {slurm_key}={found[slurm_key]}")
    if "RealMemory" in found and int(n["real_memory_mb"]) > int(found["RealMemory"]):
        problems.append(
            f"real_memory_mb = {n['real_memory_mb']} is more than the {found['RealMemory']} MB "
            "slurmd reports; Slurm would drain the node (Low RealMemory)"
        )
    if problems:
        suggestion = (
            f"cpus = {found.get('CPUs')}, sockets = {found.get('SocketsPerBoard')}, "
            f"cores_per_socket = {found.get('CoresPerSocket')}, threads_per_core = {found.get('ThreadsPerCore')}, "
            f"real_memory_mb = {int(int(found.get('RealMemory', 0)) * 0.9)}"
        )
        ctx.fail_or_warn(
            "node hardware does not match cluster.toml:\n  - " + "\n  - ".join(problems)
            + f"\nSuggested [[nodes]] values for {n['name']}: {suggestion}\n"
            "Fix cluster.toml, run `install.sh apply` on the controller, then rerun this node."
        )
    else:
        ok("CPU topology and memory match")


# ---------------------------------------------------------------------------
# Hostname and /etc/hosts
# ---------------------------------------------------------------------------


def hosts(ctx: Ctx) -> None:
    header("Hostname and /etc/hosts")
    name = ctx.hostname
    current = out(ctx, ["hostname", "-s"])
    if current != name:
        run(ctx, ["hostnamectl", "set-hostname", name])
    else:
        ok(f"hostname is {name}")
    write_file(ctx, "/etc/cloud/cloud.cfg.d/99-team7.cfg", render.cloud_cfg())
    existing = Path("/etc/hosts").read_text() if Path("/etc/hosts").exists() else ""
    write_file(ctx, "/etc/hosts", render.merge_hosts(existing, render.hosts_block(ctx.cfg)))
    names = {ctx.cfg["controller"]["name"], *(n["name"] for n in ctx.cfg["nodes"])}
    in_block = False
    for line in existing.splitlines():
        if line.strip() == render.HOSTS_BEGIN:
            in_block = True
        elif line.strip() == render.HOSTS_END:
            in_block = False
        elif not in_block and not line.lstrip().startswith("#"):
            fields = line.split()
            if len(fields) > 1 and names & set(fields[1:]):
                warn(f"/etc/hosts has another entry for a cluster host: {line.strip()!r}")


# ---------------------------------------------------------------------------
# Unix users
# ---------------------------------------------------------------------------


def _group(name: str):
    try:
        return grp.getgrnam(name)
    except KeyError:
        return None


def _user(name: str):
    try:
        return pwd.getpwnam(name)
    except KeyError:
        return None


def users(ctx: Ctx) -> None:
    header("Unix users and groups")
    u = ctx.cfg["unix"]

    g = _group(u["group"])
    if g is None:
        run(ctx, ["groupadd", "-g", str(u["gid"]), u["group"]])
    elif g.gr_gid != u["gid"]:
        raise InstallError(f"group {u['group']} exists with gid {g.gr_gid}, cluster.toml says {u['gid']}")
    else:
        ok(f"group {u['group']} ({u['gid']})")

    # The slurm user must exist (with the same UID everywhere) before the
    # Ubuntu packages create one with a random UID.
    sg = _group("slurm")
    if sg is None:
        run(ctx, ["groupadd", "-r", "-g", str(u["slurm_gid"]), "slurm"])
    su = _user("slurm")
    if su is None:
        run(ctx, ["useradd", "-r", "-u", str(u["slurm_uid"]), "-g", "slurm", "-d", "/nonexistent",
                  "-s", "/usr/sbin/nologin", "-c", "Slurm workload manager", "slurm"])
    elif su.pw_uid != u["slurm_uid"]:
        warn(f"user slurm already has uid {su.pw_uid}, not {u['slurm_uid']}; keep it the same on all hosts")
    else:
        ok(f"user slurm ({u['slurm_uid']})")

    for user in ctx.cfg["users"]:
        _ensure_user(ctx, user, u)


def _ensure_user(ctx: Ctx, user: dict, unix: dict) -> None:
    name, uid = user["name"], user["uid"]
    existing = _user(name)
    extra_groups = [g for g in user.get("groups") or [] if g != unix["group"]]
    for g in extra_groups:
        if _group(g) is None:
            raise InstallError(f"user {name}: group {g!r} in [[users]].groups does not exist on this host")
    if existing is None:
        cmd = ["useradd", "-m", "-u", str(uid), "-g", unix["group"], "-s", unix["shell"]]
        if extra_groups:
            cmd += ["-G", ",".join(extra_groups)]
        run(ctx, cmd + [name])
        home = Path(f"/home/{name}")
    else:
        if existing.pw_uid != uid:
            raise InstallError(
                f"user {name} exists with uid {existing.pw_uid}, cluster.toml says {uid}. "
                "UIDs must match on every host; fix by hand."
            )
        if existing.pw_gid != unix["gid"]:
            run(ctx, ["usermod", "-g", unix["group"], name])
        missing = [g for g in extra_groups if name not in _group(g).gr_mem]
        if missing:
            run(ctx, ["usermod", "-a", "-G", ",".join(missing), name])
        home = Path(existing.pw_dir)
        ok(f"user {name} ({uid})")
    keys = user.get("ssh_keys") or []
    if keys:
        ssh = home / ".ssh"
        ensure_dir(ctx, ssh, 0o700, name, unix["group"])
        write_file(
            ctx,
            ssh / "authorized_keys",
            "# Managed by the Team 7 installer (cluster.toml [[users]].ssh_keys)\n" + "\n".join(keys) + "\n",
            mode=0o600,
            owner=name,
            group=unix["group"],
        )


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------


def packages(ctx: Ctx) -> None:
    header("Packages")
    cfg = ctx.cfg
    pkgs = ["munge", "slurm-client", "chrony", "rsync", "curl", "ca-certificates"]
    if ctx.is_controller:
        pkgs += ["slurmctld", "slurmdbd", "mariadb-server"]
    else:
        pkgs += ["slurmd", "xfsprogs", "nvme-cli"]
    if cfg["storage"]["shared"]["type"] in ("efs", "nfs"):
        pkgs.append("nfs-common")
    if cfg["code"]["source"] == "git":
        pkgs.append("git")
    apt_install(ctx, pkgs)

    if cfg["secrets"]["munge_key_source"] == "ssm" and shutil.which("aws") is None:
        run(ctx, ["snap", "install", "aws-cli", "--classic"], timeout=600)

    want = cfg["cluster"]["slurm_version"]
    if want:
        binary = "slurmctld" if ctx.is_controller else "slurmd"
        version = out(ctx, [binary, "-V"])  # "slurm 23.11.4"
        got = version.split()[-1] if version else ""
        if not got:
            ctx.fail_or_warn(f"could not read the {binary} version")
        elif not got.startswith(want):
            ctx.fail_or_warn(
                f"{binary} is {got}, cluster.toml wants {want}. All hosts must run the same "
                "Slurm version; update [cluster].slurm_version or the OS image."
            )
        else:
            ok(f"Slurm {got}")


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def time_sync(ctx: Ctx) -> None:
    header("Time sync (munge needs it)")
    if write_file(ctx, "/etc/chrony/sources.d/team7.sources", render.chrony_sources(ctx.cfg)):
        restart(ctx, "chrony")
    else:
        ok("chrony source configured")
    tz = ctx.cfg["cluster"]["timezone"]
    if tz and out(ctx, ["timedatectl", "show", "-p", "Timezone", "--value"]) != tz:
        run(ctx, ["timedatectl", "set-timezone", tz])


# ---------------------------------------------------------------------------
# Shared storage (EFS / NFS)
# ---------------------------------------------------------------------------

FSTAB_BEGIN = "# BEGIN team7 shared storage (managed by the Team 7 installer)"
FSTAB_END = "# END team7 shared storage"


def shared_storage(ctx: Ctx) -> None:
    sh = ctx.cfg["storage"]["shared"]
    header(f"Shared storage ({sh['type']})")
    fstab = Path("/etc/fstab")
    existing = fstab.read_text() if fstab.exists() else ""
    if sh["type"] == "none":
        if FSTAB_BEGIN in existing:
            warn("/etc/fstab still has a team7 shared mount; remove it by hand if no longer used")
        ok("no shared storage configured")
        return
    if sh["type"] == "efs":
        source = f"{sh['efs_id']}.efs.{sh['region']}.amazonaws.com:/"
        opts = sh["options"] or "nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev"
    else:
        source = f"{sh['nfs_server']}:{sh['nfs_export']}"
        opts = sh["options"] or "nfsvers=4.1,hard,timeo=600,retrans=2,_netdev"
    line = f"{source} {sh['mount']} nfs4 {opts} 0 0"
    block = f"{FSTAB_BEGIN}\n{line}\n{FSTAB_END}"
    out_lines, skip = [], False
    for ln in existing.splitlines():
        if ln.strip() == FSTAB_BEGIN:
            skip = True
            continue
        if skip and ln.strip() == FSTAB_END:
            skip = False
            continue
        if not skip:
            out_lines.append(ln)
    write_file(ctx, fstab, "\n".join(out_lines + [block]) + "\n")
    ensure_dir(ctx, sh["mount"])
    if out(ctx, ["findmnt", "-n", "-o", "SOURCE", sh["mount"]]) != source:
        systemctl(ctx, "daemon-reload")
        run(ctx, ["mount", sh["mount"]], timeout=120)
    ok(f"{source} mounted on {sh['mount']}")
    ck = ctx.cfg["checkpoints"]
    if ck["durable"] == "shared" and ck["shared_dir"]:
        ensure_dir(ctx, ck["shared_dir"], 0o2775, "root", ctx.cfg["unix"]["group"])
    logs = ctx.cfg["logs"]["dir"]
    if logs.startswith(sh["mount"].rstrip("/") + "/"):
        ensure_dir(ctx, logs, 0o1777)


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


def code(ctx: Ctx) -> None:
    header("Code")
    # Always install a copy of the cluster/ directory (installer, team7config,
    # pyenv, tests) at /opt/team7/cluster so it is the same on every host.
    dest_cluster = render.CLUSTER_COPY
    if ctx.cluster_dir.resolve() != dest_cluster.resolve():
        ensure_dir(ctx, dest_cluster.parent)
        run(ctx, ["rsync", "-a", "--delete", "--exclude", "__pycache__", "--exclude", ".venv",
                  "--exclude", "out/", "--exclude", "evidence/", "--chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r",
                  f"{ctx.cluster_dir}/", f"{dest_cluster}/"])
        for script in ("install.sh", "tests/test_limits.sh", "tests/collect_evidence.sh"):
            p = dest_cluster / script
            if p.exists() and not ctx.dry_run:
                os.chmod(p, 0o755)
    ok(f"cluster tools at {dest_cluster}")

    c = ctx.cfg["code"]
    if c["source"] == "none":
        ok("[code].source = none, no code deployed")
        return
    dest = Path(c["dest"])
    if c["source"] == "local":
        src = (ctx.cluster_dir / c["local_path"]).resolve()
        if not src.is_dir():
            raise InstallError(f"[code].local_path resolves to {src}, which is not a directory")
        if src == dest.resolve():
            ok(f"code already at {dest}")
            return
        excludes = []
        for e in c["exclude"]:
            excludes += ["--exclude", e]
        ensure_dir(ctx, dest.parent)
        run(ctx, ["rsync", "-a", "--delete", *excludes, "--chmod=Dgo+rx,Fgo+r", f"{src}/", f"{dest}/"])
        ok(f"copied {src} to {dest}")
    else:
        if not (dest / ".git").is_dir():
            ensure_dir(ctx, dest.parent)
            run(ctx, ["git", "clone", "--no-checkout", c["git_url"], str(dest)], timeout=900)
        run(ctx, ["git", "-C", str(dest), "fetch", "--tags", "origin", c["git_ref"]], timeout=900)
        run(ctx, ["git", "-C", str(dest), "checkout", "--force", "--detach", "FETCH_HEAD"])
        commit = out(ctx, ["git", "-C", str(dest), "rev-parse", "HEAD"])
        ok(f"{c['git_url']} @ {c['git_ref']} ({commit[:12]}) in {dest}")


# ---------------------------------------------------------------------------
# Installed TOML, environment, job templates
# ---------------------------------------------------------------------------


def config_files(ctx: Ctx) -> None:
    header("cluster.toml, environment and job templates")
    src = Path(ctx.cfg.path)
    data = src.read_bytes()
    ensure_dir(ctx, ETC, 0o755)
    write_file(ctx, INSTALLED_TOML, data, mode=0o644)
    write_file(ctx, ETC / "cluster.toml.sha256", file_sha256(src) + "  cluster.toml\n", mode=0o644)
    write_file(ctx, render.ENV_FILE, render.team7_env(ctx.cfg))
    write_file(ctx, "/etc/profile.d/team7.sh", render.profile_sh())
    write_file(ctx, "/usr/local/bin/team7-config", render.team7_config_wrapper(), mode=0o755)
    ensure_dir(ctx, render.JOBS_DIR)
    write_file(ctx, render.JOBS_DIR / "train.sbatch", render.train_sbatch(ctx.cfg), mode=0o755)
    write_file(ctx, render.JOBS_DIR / "eval.sbatch", render.eval_sbatch(ctx.cfg), mode=0o755)
    write_file(ctx, render.JOBS_DIR / "debug.sbatch", render.debug_sbatch(ctx.cfg), mode=0o755)
    ok(f"installed {INSTALLED_TOML} (sha256 {file_sha256(src)[:16]}...)")
