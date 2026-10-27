"""Shared helpers: command execution, idempotent file writes, logging."""

from __future__ import annotations

import difflib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path("/opt/team7")
ETC = Path("/etc/team7")
STATE_DIR = ETC / "state"
INSTALLED_TOML = ETC / "cluster.toml"


class InstallError(Exception):
    """A step failed. The message says what to fix."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def header(title: str) -> None:
    print(_c("1;34", f"\n==> {title}"), flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def ok(msg: str) -> None:
    print(_c("32", f"    ok: {msg}"), flush=True)


def warn(msg: str) -> None:
    print(_c("33", f"    warning: {msg}"), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    cfg: Any                       # team7config.Config
    cluster_dir: Path              # the cluster/ directory the installer runs from
    role: str                      # "controller" | "node" | "none"
    node: dict | None = None       # the [[nodes]] entry when role == "node"
    dry_run: bool = False
    munge_key: Path | None = None  # --munge-key
    force: bool = False
    changed: list[str] = field(default_factory=list)
    # Set by steps and read by later steps in the same run.
    restart_ctld: bool = False
    reconfigure_ctld: bool = False
    node_defs_changed: bool = False
    db_password: str | None = None
    _apt_updated: bool = False

    @property
    def hostname(self) -> str:
        if self.role == "controller":
            return self.cfg["controller"]["name"]
        if self.node:
            return self.node["name"]
        return os.uname().nodename.split(".")[0]

    @property
    def is_controller(self) -> bool:
        return self.role == "controller"

    @property
    def is_node(self) -> bool:
        return self.role == "node"

    def fail_or_warn(self, msg: str) -> None:
        """Checks fail hard on a real run and only warn in --dry-run."""
        if self.dry_run:
            warn(f"(dry-run, would fail) {msg}")
        else:
            raise InstallError(msg)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def fmt_cmd(cmd: list[str] | str) -> str:
    return cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(c)) for c in cmd)


def run(
    ctx: Ctx,
    cmd: list[str] | str,
    *,
    check: bool = True,
    capture: bool = True,
    input: str | bytes | None = None,
    env: dict[str, str] | None = None,
    readonly: bool = False,
    timeout: float | None = None,
    cwd: str | os.PathLike | None = None,
    quiet: bool = False,
    secret: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command.

    readonly=True marks commands that only inspect state; they also run in
    --dry-run. Everything else is printed, not executed, in --dry-run.
    secret=True hides stdin/command details from logs.
    """
    shown = "<command hidden>" if secret else fmt_cmd(cmd)
    if ctx.dry_run and not readonly:
        info(f"[dry-run] $ {shown}")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    exe = cmd.split()[0] if isinstance(cmd, str) else str(cmd[0])
    if ctx.dry_run and shutil.which(exe) is None:
        return subprocess.CompletedProcess(cmd, 127, "", f"{exe}: not found (dry-run)")
    if not quiet and not readonly:
        info(f"$ {shown}")
    full_env = None
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    text = not isinstance(input, bytes)
    try:
        cp = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            input=input,
            env=full_env,
            capture_output=capture,
            text=text,
            timeout=timeout,
            cwd=cwd,
        )
    except FileNotFoundError:
        if check:
            raise InstallError(f"command not found: {exe}") from None
        return subprocess.CompletedProcess(cmd, 127, "", f"{exe}: not found")
    except subprocess.TimeoutExpired:
        raise InstallError(f"timed out after {timeout}s: {shown}") from None
    if check and cp.returncode != 0:
        tail = ""
        if capture:
            out = (cp.stderr or "") + (cp.stdout or "")
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            tail = "\n" + "\n".join(out.strip().splitlines()[-20:])
        raise InstallError(f"command failed ({cp.returncode}): {shown}{tail}")
    return cp


def out(ctx: Ctx, cmd: list[str] | str, **kw) -> str:
    """stdout of a read-only command ('' if it fails)."""
    cp = run(ctx, cmd, check=False, readonly=True, **kw)
    return (cp.stdout or "").strip() if cp.returncode == 0 else ""


def systemctl(ctx: Ctx, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(ctx, ["systemctl", *args], check=check)


def daemon_reload(ctx: Ctx) -> None:
    systemctl(ctx, "daemon-reload")


def is_active(ctx: Ctx, unit: str) -> bool:
    return out(ctx, ["systemctl", "is-active", unit]) == "active"


def restart(ctx: Ctx, unit: str, enable: bool = True) -> None:
    if enable:
        systemctl(ctx, "enable", unit)
    systemctl(ctx, "restart", unit)


def wait_for(
    ctx: Ctx, predicate: Callable[[], bool], what: str, timeout: float = 60, interval: float = 2
) -> None:
    if ctx.dry_run:
        info(f"[dry-run] would wait for {what}")
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise InstallError(f"timed out after {timeout:.0f}s waiting for {what}")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _chown(path: Path, owner: str, group: str) -> None:
    try:
        shutil.chown(path, user=owner, group=group)
    except (LookupError, PermissionError) as ex:
        raise InstallError(f"chown {owner}:{group} {path}: {ex}") from None


def write_file(
    ctx: Ctx,
    path: str | os.PathLike,
    content: str | bytes,
    mode: int = 0o644,
    owner: str = "root",
    group: str = "root",
    backup: bool = True,
    secret: bool = False,
) -> bool:
    """Write a file only if content or mode differ. Returns True if changed."""
    p = Path(path)
    data = content.encode() if isinstance(content, str) else content
    try:
        old = p.read_bytes() if p.exists() else None
        same_mode = old is not None and (p.stat().st_mode & 0o7777) == mode
    except PermissionError:
        if not ctx.dry_run:
            raise
        old, same_mode = None, False
    same_content = old == data
    if same_content and same_mode:
        return False
    if ctx.dry_run:
        if same_content:
            info(f"[dry-run] chmod {oct(mode)} {p}")
        elif secret or isinstance(content, bytes):
            info(f"[dry-run] would write {p} ({len(data)} bytes, mode {oct(mode)})")
        else:
            info(f"[dry-run] would write {p} (mode {oct(mode)})")
            old_lines = old.decode(errors="replace").splitlines() if old else []
            diff = list(
                difflib.unified_diff(old_lines, content.splitlines(), str(p), str(p), lineterm="", n=1)
            )
            for line in diff[:60]:
                info(f"    {line}")
            if len(diff) > 60:
                info(f"    ... {len(diff) - 60} more diff lines")
        return True
    p.parent.mkdir(parents=True, exist_ok=True)
    if old is not None and not same_content and backup:
        bak = p.with_name(p.name + ".team7.bak")
        shutil.copy2(p, bak)
        os.chmod(bak, mode & 0o700 if secret else mode)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        _chown(Path(tmp), owner, group)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    info(f"wrote {p}")
    ctx.changed.append(str(p))
    return True


def ensure_dir(
    ctx: Ctx, path: str | os.PathLike, mode: int = 0o755, owner: str = "root", group: str = "root"
) -> None:
    p = Path(path)
    if ctx.dry_run:
        try:
            exists = p.is_dir()
        except PermissionError:
            exists = False
        if not exists:
            info(f"[dry-run] mkdir -p {p} (mode {oct(mode)}, {owner}:{group})")
        return
    p.mkdir(parents=True, exist_ok=True)
    if (p.stat().st_mode & 0o7777) != mode:
        os.chmod(p, mode)
    _chown(p, owner, group)


def remove_file(ctx: Ctx, path: str | os.PathLike) -> None:
    p = Path(path)
    if not p.exists():
        return
    if ctx.dry_run:
        info(f"[dry-run] rm {p}")
        return
    p.unlink()
    info(f"removed {p}")


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------


def apt_installed(ctx: Ctx, pkg: str) -> bool:
    return "install ok installed" in out(ctx, ["dpkg-query", "-W", "-f=${Status}", pkg])


def apt_install(ctx: Ctx, packages: list[str]) -> None:
    missing = [p for p in packages if not apt_installed(ctx, p)]
    if not missing:
        ok(f"packages present: {' '.join(packages)}")
        return
    env = {"DEBIAN_FRONTEND": "noninteractive", "NEEDRESTART_MODE": "a"}
    if not ctx._apt_updated:
        run(ctx, ["apt-get", "update", "-q"], env=env, timeout=600)
        ctx._apt_updated = True
    run(
        ctx,
        ["apt-get", "install", "-y", "-q", "--no-install-recommends", *missing],
        env=env,
        timeout=1800,
    )


# ---------------------------------------------------------------------------
# State markers
# ---------------------------------------------------------------------------


def read_text_or_none(path: str | os.PathLike) -> str | None:
    """File content, or None if missing or unreadable (non-root --dry-run)."""
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return None


def read_state(name: str) -> str | None:
    text = read_text_or_none(STATE_DIR / name)
    return text.strip() if text is not None else None


def write_state(ctx: Ctx, name: str, value: str) -> None:
    ensure_dir(ctx, STATE_DIR, 0o700)
    write_file(ctx, STATE_DIR / name, value + "\n", mode=0o600, backup=False)
