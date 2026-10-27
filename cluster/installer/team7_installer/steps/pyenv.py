"""Training environment on GPU nodes: pinned uv, uv-managed Python, a venv
built from pyenv/uv.lock (torch + DeepSpeed), and pre-built DeepSpeed ops."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import tempfile
from pathlib import Path

from team7config import file_sha256

from .. import render
from ..util import (
    Ctx,
    InstallError,
    ensure_dir,
    header,
    info,
    ok,
    out,
    run,
    warn,
    write_state,
)

UV_BIN = Path("/usr/local/bin/uv")


def _free_mb(path: str) -> int | None:
    """Free megabytes on the filesystem holding path, or None if it doesn't exist yet."""
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return int(st.f_bavail * st.f_frsize / (1024 * 1024))


def _uv_env(ctx: Ctx) -> dict[str, str]:
    py = ctx.cfg["python"]
    return {
        "UV_PYTHON_INSTALL_DIR": py["python_install_dir"],
        "UV_PROJECT_ENVIRONMENT": py["venv"],
        "UV_CACHE_DIR": py["cache_dir"],
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON": py["version"],
        "UV_LINK_MODE": "copy",
        "UV_NO_PROGRESS": "1",
        "CUDA_HOME": py["cuda_home"],
        "TORCH_CUDA_ARCH_LIST": py["cuda_arch_list"],
        "TORCH_EXTENSIONS_DIR": py["torch_extensions_dir"],
        "PATH": f"{py['cuda_home']}/bin:/usr/local/bin:/usr/bin:/bin",
    }


def _install_uv(ctx: Ctx) -> None:
    py = ctx.cfg["python"]
    want = py["uv_version"]
    have = out(ctx, [str(UV_BIN), "--version"]) if UV_BIN.exists() else ""
    if have.split()[1:2] == [want]:
        ok(f"uv {want}")
        return
    arch = {"x86_64": "x86_64", "aarch64": "aarch64"}.get(platform.machine())
    if not arch:
        raise InstallError(f"unsupported CPU architecture {platform.machine()}")
    name = f"uv-{arch}-unknown-linux-gnu"
    url = f"https://github.com/astral-sh/uv/releases/download/{want}/{name}.tar.gz"
    if ctx.dry_run:
        info(f"[dry-run] would download {url} and install {UV_BIN}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "uv.tar.gz"
        run(ctx, ["curl", "-fsSL", "--retry", "3", "-o", str(tar), url], timeout=300)
        if py["uv_sha256"]:
            got = hashlib.sha256(tar.read_bytes()).hexdigest()
            if got != py["uv_sha256"]:
                raise InstallError(f"uv download sha256 {got} does not match [python].uv_sha256")
        run(ctx, ["tar", "-xzf", str(tar), "-C", tmp])
        for binary in ("uv", "uvx"):
            run(ctx, ["install", "-m", "0755", str(Path(tmp) / name / binary), f"/usr/local/bin/{binary}"])
    ok(f"installed uv {want}")


def pyenv(ctx: Ctx) -> None:
    header("Python training environment (uv, torch, DeepSpeed)")
    py = ctx.cfg["python"]
    group = ctx.cfg["unix"]["group"]
    project = render.CLUSTER_COPY / py["project"]
    if not (project / "pyproject.toml").exists() and not ctx.dry_run:
        raise InstallError(f"{project}/pyproject.toml not found ([python].project = {py['project']!r})")

    _install_uv(ctx)
    ensure_dir(ctx, py["python_install_dir"], 0o755)
    ensure_dir(ctx, py["cache_dir"], 0o700)
    ensure_dir(ctx, Path(py["venv"]).parent, 0o755)
    ensure_dir(ctx, py["torch_extensions_dir"], 0o2775, "root", group)
    env = _uv_env(ctx)
    uv = str(UV_BIN)

    run(ctx, [uv, "python", "install", py["version"]], env=env, timeout=900)
    lock = project / "uv.lock"
    if not lock.exists():
        warn(f"{lock} missing: resolving now. Copy it back into the repo and commit it so every node matches.")
        run(ctx, [uv, "lock"], env=env, cwd=project, timeout=1800)
    # DeepSpeed builds without isolation and imports torch in setup.py, so torch
    # must be in the venv first.
    run(ctx, [uv, "sync", "--frozen", "--no-install-package", "deepspeed"], env=env, cwd=project, timeout=3600)
    run(ctx, [uv, "sync", "--frozen"], env=env, cwd=project, timeout=3600)
    run(ctx, ["chmod", "-R", "a+rX", py["venv"], py["python_install_dir"]])

    python = f"{py['venv']}/bin/python"
    if ctx.dry_run:
        info("[dry-run] would check torch/CUDA and pre-build DeepSpeed ops")
        return

    # The two-pass sync can occasionally leave a large wheel (e.g. nvidia-cudnn-cu12,
    # ~500 MB) half-materialized: the package dir exists but its .so files are missing,
    # so `import torch` fails with "libcudnn.so.9: cannot open shared object file".
    # Verify the import and, if it fails, reinstall every package once to self-heal.
    check = run(ctx, [python, "-c", "import torch"], env=env, check=False, timeout=300, quiet=True)
    if check.returncode != 0:
        err = (check.stderr or check.stdout or "").strip().splitlines()
        warn("import torch failed after sync (" + (err[-1] if err else "?") + "); reinstalling packages")
        free_mb = _free_mb(py["venv"])
        if free_mb is not None and free_mb < 5000:
            raise InstallError(
                f"only {free_mb} MB free on the filesystem holding {py['venv']}. The CUDA wheels "
                "need several GB; a full disk truncates them. Free space or use a larger root "
                "volume, then rerun. (The DLAMI root volume should be >= 100 GB.)"
            )
        run(ctx, [uv, "sync", "--frozen", "--reinstall"], env=env, cwd=project, timeout=3600)
        run(ctx, ["chmod", "-R", "a+rX", py["venv"], py["python_install_dir"]])
        check = run(ctx, [python, "-c", "import torch"], env=env, check=False, timeout=300, quiet=True)
        if check.returncode != 0:
            err = (check.stderr or check.stdout or "").strip().splitlines()
            raise InstallError("import torch still fails after a full reinstall: " + (err[-1] if err else "?"))
        ok("reinstall fixed the environment")

    torch_cuda = out(ctx, [python, "-c", "import torch; print(torch.version.cuda or '')"], env=env)
    if not torch_cuda:
        raise InstallError("torch in the venv has no CUDA build; check the torch index in pyproject.toml")
    probe = out(ctx, [python, "-c",
                      "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), "
                      "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"], env=env)
    if not probe.startswith("True"):
        raise InstallError(f"torch cannot use the GPU ({probe or 'no output'}); check the driver vs CUDA {torch_cuda}")
    ok(f"torch CUDA {torch_cuda}: {probe}")

    ops = py["deepspeed_prebuild_ops"]
    if ops:
        nvcc = Path(py["cuda_home"]) / "bin" / "nvcc"
        version = out(ctx, [str(nvcc), "--version"])
        m = re.search(r"release (\d+)\.(\d+)", version)
        if not m:
            raise InstallError(
                f"{nvcc} not found or unreadable. DeepSpeed compiles ops with nvcc; set "
                "[python].cuda_home to a CUDA toolkit, or deepspeed_prebuild_ops = []"
            )
        if m.group(1) != torch_cuda.split(".")[0]:
            raise InstallError(
                f"nvcc is CUDA {m.group(1)}.{m.group(2)} but torch is built for CUDA {torch_cuda}; "
                "DeepSpeed refuses a major-version mismatch. Point [python].cuda_home at a "
                f"CUDA {torch_cuda.split('.')[0]}.x toolkit (e.g. /usr/local/cuda-{torch_cuda})."
            )
        code = (
            "import importlib\n"
            "b = importlib.import_module('deepspeed.ops.op_builder')\n"
            f"for name in {ops!r}:\n"
            "    print('building', name, flush=True)\n"
            "    getattr(b, name)().load()\n"
        )
        run(ctx, [python, "-c", code], env=env, timeout=3600)
        run(ctx, ["chown", "-R", f"root:{group}", py["torch_extensions_dir"]])
        run(ctx, ["chmod", "-R", "g+rwX,o+rX", py["torch_extensions_dir"]])
        run(ctx, ["find", py["torch_extensions_dir"], "-type", "d", "-exec", "chmod", "g+s", "{}", "+"])
        ok(f"pre-built DeepSpeed ops: {', '.join(ops)}")

    report = run(ctx, [f"{py['venv']}/bin/ds_report"], env=env, check=False, timeout=300, quiet=True)
    if report.returncode == 0:
        for line in (report.stdout or "").splitlines():
            if any(k in line for k in ("torch version", "deepspeed info", "torch cuda version", "nvcc version")):
                info(line.strip())
    else:
        warn("ds_report failed; run it by hand to see why")
    write_state(ctx, "pyenv.lock.sha256", file_sha256(lock) if lock.exists() else "none")
