"""GPU-node-only steps: scratch disk, health hooks, configless slurmd."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .. import render
from ..util import (
    Ctx,
    InstallError,
    daemon_reload,
    ensure_dir,
    header,
    info,
    is_active,
    ok,
    out,
    restart,
    run,
    systemctl,
    wait_for,
    warn,
    write_file,
)


def scratch(ctx: Ctx) -> None:
    sc = ctx.cfg["storage"]["scratch"]
    header(f"Scratch disk ({sc['type']} on {sc['mount']})")
    changed = write_file(ctx, render.SCRATCH_SCRIPT, render.scratch_setup_sh(ctx.cfg), mode=0o755)
    changed |= write_file(ctx, "/etc/systemd/system/team7-scratch.service", render.scratch_unit())
    if changed:
        daemon_reload(ctx)
    systemctl(ctx, "enable", "team7-scratch.service")
    # oneshot + RemainAfterExit: restart reruns the (idempotent) script.
    cp = systemctl(ctx, "restart", "team7-scratch.service", check=False)
    if cp.returncode != 0 and not ctx.dry_run:
        logs = out(ctx, ["journalctl", "-u", "team7-scratch.service", "-n", "20", "--no-pager"])
        raise InstallError(f"team7-scratch.service failed:\n{logs}")
    if not ctx.dry_run:
        if sc["type"] == "nvme":
            src = out(ctx, ["findmnt", "-n", "-o", "SOURCE", sc["mount"]])
            if not src:
                raise InstallError(f"{sc['mount']} is not mounted after team7-scratch.service ran")
            ok(f"{src} mounted on {sc['mount']} (instance store: erased on stop/start, kept on reboot)")
        else:
            # type = "dir": the mount is owned by the AMI (e.g. DLAMI /opt/dlami/nvme).
            fs = out(ctx, ["df", "-P", sc["mount"]]).splitlines()
            where = fs[-1].split()[-1] if len(fs) > 1 else "?"
            mp = out(ctx, ["findmnt", "-n", "-o", "TARGET", "--target", sc["mount"]])
            ok(f"using directory {sc['mount']} on filesystem mounted at {mp or where} "
               "(instance store if on ephemeral NVMe: erased on stop/start)")
            if mp == "/" :
                warn(f"{sc['mount']} is on the root filesystem, not a separate fast disk; "
                     "check that the AMI's NVMe is mounted, or set a path under it")
    else:
        ok(f"{sc['mount']} ready")


def health(ctx: Ctx) -> None:
    h = ctx.cfg["health"]
    header("Health hooks (Prolog / HealthCheckProgram)")
    if not h["enabled"]:
        ok("[health].enabled = false; slurm.conf has no Prolog or HealthCheckProgram")
        return
    launch = Path(h["launch_script"])
    if not ctx.dry_run and not (launch.is_file() and os.access(launch, os.X_OK)):
        raise InstallError(
            f"[health].launch_script {launch} is missing or not executable. Deploy health/ with "
            "[code], or set [health].enabled = false until it exists. Without it the Prolog "
            "fails and Slurm drains this node on the first job."
        )
    ensure_dir(ctx, render.SLURM_SCRIPTS)
    ensure_dir(ctx, h["log_dir"])
    write_file(ctx, render.PROLOG, render.prolog_sh(ctx.cfg), mode=0o755)
    write_file(ctx, render.HEALTHCHECK, render.healthcheck_sh(ctx.cfg), mode=0o755)
    if not ctx.dry_run:
        cp = run(ctx, [str(launch), *h["quick_args"].split()], check=False, timeout=h["check_timeout_s"])
        if cp.returncode != 0:
            raise InstallError(
                f"health check fails on this node (rc={cp.returncode}); the node would be drained.\n"
                f"{(cp.stderr or cp.stdout or '').strip()[-2000:]}"
            )
        ok("quick health check passes")


def slurmd(ctx: Ctx) -> None:
    header("slurmd (configless)")
    s = ctx.cfg["slurm"]
    name = ctx.node["name"]
    ensure_dir(ctx, s["slurmd_spool_dir"], 0o755)
    ensure_dir(ctx, s["log_dir"], 0o755, "slurm", "slurm")
    changed = write_file(ctx, "/etc/default/slurmd", render.slurmd_default(ctx.cfg))
    changed |= write_file(ctx, "/etc/systemd/system/slurmd.service.d/team7.conf", render.slurmd_dropin())
    local_conf = Path(s["conf_dir"]) / "slurm.conf"
    if local_conf.exists():
        # A local slurm.conf would shadow the controller's copy for client commands.
        if ctx.dry_run:
            info(f"[dry-run] would move {local_conf} aside")
        else:
            local_conf.rename(local_conf.with_name("slurm.conf.team7-disabled"))
            warn(f"moved {local_conf} aside; this node gets its config from the controller")
        changed = True
    if changed:
        daemon_reload(ctx)
    if changed or not is_active(ctx, "slurmd") or ctx.force:
        restart(ctx, "slurmd")
    else:
        ok("slurmd running, unchanged")
    if ctx.dry_run:
        return

    args = out(ctx, ["ps", "-o", "args=", "-C", "slurmd"])
    if "--conf-server" not in args:
        raise InstallError(
            "slurmd is running without --conf-server, so the systemd unit ignores "
            "/etc/default/slurmd. Check `systemctl cat slurmd` and add the option to ExecStart "
            "in /etc/systemd/system/slurmd.service.d/team7.conf."
        )

    state = {"line": ""}

    def registered() -> bool:
        state["line"] = out(ctx, ["scontrol", "-o", "show", "node", name])
        st = re.search(r"\bState=(\S+)", state["line"])
        return bool(st) and "UNKNOWN" not in st.group(1) and "NOT_RESPONDING" not in st.group(1)

    try:
        wait_for(ctx, registered, f"{name} to register with the controller", timeout=120, interval=3)
    except InstallError:
        logs = out(ctx, ["journalctl", "-u", "slurmd", "-n", "30", "--no-pager"])
        raise InstallError(
            f"{name} did not register. Check that the controller is up, that ports "
            f"{s['ctld_port']}/{s['slurmd_port']} are open between hosts, and the munge key "
            f"matches.\nslurmd log:\n{logs}"
        ) from None
    line = state["line"]
    st = re.search(r"\bState=(\S+)", line).group(1)
    reason = re.search(r"\bReason=(.*?)(?: \w+=|$)", line)
    if "DRAIN" in st or "DOWN" in st:
        warn(f"{name} is {st}" + (f": {reason.group(1)}" if reason else ""))
        if reason and "Low" in reason.group(1):
            raise InstallError(
                f"Slurm drained {name} because the hardware is smaller than cluster.toml says "
                f"({reason.group(1)}). Lower cpus/real_memory_mb, run `install.sh apply` on the "
                "controller, then rerun this step."
            )
        info(f"after fixing the cause: sudo scontrol update nodename={name} state=resume")
    else:
        ok(f"{name} registered, state {st}")
