"""Munge key: the only secret shared between hosts.

file mode: the controller generates the key and writes it to
[secrets].munge_key_file. You copy it to each node and pass --munge-key.

ssm mode: the controller stores the key (base64) as an SSM SecureString;
nodes read it with their instance role.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from pathlib import Path

from ..util import (
    Ctx,
    InstallError,
    ensure_dir,
    header,
    info,
    ok,
    read_state,
    restart,
    run,
    write_file,
    write_state,
)

MUNGE_KEY = Path("/etc/munge/munge.key")
KEY_BYTES = 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ssm_get(ctx: Ctx) -> bytes | None:
    s = ctx.cfg["secrets"]
    cp = run(
        ctx,
        ["aws", "ssm", "get-parameter", "--with-decryption", "--name", s["ssm_parameter"],
         "--region", s["ssm_region"], "--query", "Parameter.Value", "--output", "text"],
        check=False,
        readonly=True,
        secret=True,
    )
    if cp.returncode != 0:
        if "ParameterNotFound" in (cp.stderr or ""):
            return None
        if ctx.dry_run:
            return None
        raise InstallError(f"reading SSM parameter {s['ssm_parameter']} failed: {(cp.stderr or '').strip()}")
    return base64.b64decode(cp.stdout.strip())


def _ssm_put(ctx: Ctx, key: bytes) -> None:
    s = ctx.cfg["secrets"]
    if ctx.dry_run:
        info(f"[dry-run] would store the munge key in SSM {s['ssm_parameter']}")
        return
    fd, tmp = tempfile.mkstemp(prefix="team7-munge-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(base64.b64encode(key).decode())
        os.chmod(tmp, 0o600)
        run(
            ctx,
            ["aws", "ssm", "put-parameter", "--type", "SecureString", "--name", s["ssm_parameter"],
             "--region", s["ssm_region"], "--value", f"file://{tmp}"],
            secret=True,
        )
    finally:
        os.unlink(tmp)
    ok(f"stored munge key in SSM {s['ssm_parameter']}")


def _read(ctx: Ctx, path: Path) -> bytes | None:
    """File content, or None if missing (or unreadable during a non-root --dry-run)."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except PermissionError:
        if ctx.dry_run:
            return None
        raise InstallError(f"cannot read {path}; run as root") from None


def munge(ctx: Ctx) -> None:
    header("Munge key")
    s = ctx.cfg["secrets"]
    key_file = Path(s["munge_key_file"])
    installed = _read(ctx, MUNGE_KEY)
    from_file = _read(ctx, key_file)
    team_sha = read_state("munge.sha256")
    already_team_key = installed is not None and team_sha == _sha(installed)
    key: bytes | None = None

    if ctx.is_controller:
        if s["munge_key_source"] == "ssm":
            key = _ssm_get(ctx)
            if key is None:
                key = from_file or (installed if already_team_key else os.urandom(KEY_BYTES))
                _ssm_put(ctx, key)
            else:
                ok("munge key read from SSM")
        elif from_file:
            key = from_file
        elif already_team_key:
            key = installed
        else:
            key = os.urandom(KEY_BYTES)
            info("generated a new munge key")
        # Keep a copy at munge_key_file (for scp to nodes, and for restores).
        ensure_dir(ctx, key_file.parent, 0o700)
        write_file(ctx, key_file, key, mode=0o400, backup=False, secret=True)
    else:
        if ctx.munge_key:
            if not ctx.munge_key.exists():
                raise InstallError(f"--munge-key {ctx.munge_key} does not exist")
            key = ctx.munge_key.read_bytes()
        elif s["munge_key_source"] == "ssm":
            key = _ssm_get(ctx)
            if key is None and not ctx.dry_run:
                raise InstallError(
                    f"SSM parameter {s['ssm_parameter']} not found; run the controller install first"
                )
        elif from_file:
            key = from_file
        elif already_team_key:
            key = installed
        elif ctx.dry_run:
            info("[dry-run] no munge key available; a real run needs --munge-key")
            return
        else:
            raise InstallError(
                "no munge key. Copy it from the controller and pass it. From your laptop:\n"
                f"  ssh ubuntu@<controller-address> 'sudo cat {key_file}' > munge.key\n"
                "  scp munge.key ubuntu@<this-node-address>:/tmp/munge.key && rm munge.key\n"
                f"then here: sudo ./install.sh node {ctx.hostname} --munge-key /tmp/munge.key"
            )

    if key is None:
        return
    if len(key) < 32:
        raise InstallError(f"munge key is only {len(key)} bytes; expected at least 32")
    changed = write_file(ctx, MUNGE_KEY, key, mode=0o400, owner="munge", group="munge",
                         backup=False, secret=True)
    write_state(ctx, "munge.sha256", _sha(key))
    if changed or not already_team_key:
        restart(ctx, "munge")
    else:
        run(ctx, ["systemctl", "enable", "--now", "munge"], quiet=True)
    if ctx.munge_key and str(ctx.munge_key).startswith("/tmp/") and not ctx.dry_run:
        ctx.munge_key.unlink()
        info(f"deleted {ctx.munge_key}")
    cp = run(ctx, "munge -n | unmunge", check=False, readonly=True)
    if cp.returncode != 0 and not ctx.dry_run:
        raise InstallError(f"local munge test failed: {(cp.stderr or cp.stdout).strip()}")
    ok(f"munge key installed (sha256 {_sha(key)[:12]}...); same hash must appear on every host")
