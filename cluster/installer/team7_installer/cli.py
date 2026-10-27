"""install.sh entry point.

    install.sh controller                     set up ctl-01 (run first)
    install.sh node <name> [--munge-key F]    set up a GPU node
    install.sh apply                          push TOML changes on the controller
    install.sh accounts                       only reconcile QOS/accounts/users
    install.sh verify                         run a test job on every node
    install.sh render [--out DIR]             write generated files to DIR, install nothing
    install.sh validate                       check cluster.toml
    install.sh steps                          list the steps of each command
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import team7config

from . import render
from .steps import accounts as accounts_step
from .steps import common, controller, munge, node, pyenv, verify
from .util import Ctx, InstallError, header, info, ok, warn

STEPS = {
    "preflight": common.preflight,
    "hosts": common.hosts,
    "users": common.users,
    "packages": common.packages,
    "hardware": common.hardware,
    "time": common.time_sync,
    "shared": common.shared_storage,
    "code": common.code,
    "config": common.config_files,
    "munge": munge.munge,
    "slurm-config": controller.slurm_config,
    "database": controller.database,
    "slurmdbd": controller.slurmdbd,
    "accounts": accounts_step.accounts,
    "slurmctld": controller.slurmctld,
    "scratch": node.scratch,
    "health": node.health,
    "pyenv": pyenv.pyenv,
    "slurmd": node.slurmd,
    "check-controller": verify.check_controller,
    "check-node": verify.check_node,
}

PLANS = {
    "controller": ["preflight", "hosts", "users", "packages", "time", "shared", "code", "config",
                   "munge", "slurm-config", "database", "slurmdbd", "accounts", "slurmctld",
                   "check-controller"],
    "node": ["preflight", "hosts", "users", "packages", "hardware", "time", "shared", "code",
             "config", "munge", "scratch", "health", "pyenv", "slurmd", "check-node"],
    "apply": ["preflight", "hosts", "users", "time", "code", "config", "slurm-config", "accounts",
              "slurmctld", "check-controller"],
    "accounts": ["accounts"],
}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="install.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-dir", type=Path, default=Path(__file__).resolve().parents[2],
                    help=argparse.SUPPRESS)
    ap.add_argument("--config", type=Path, help="cluster TOML (default: <cluster dir>/cluster.toml)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")
    ap.add_argument("--only", help="comma-separated steps to run (see `install.sh steps`)")
    ap.add_argument("--skip", help="comma-separated steps to skip")
    ap.add_argument("--force", action="store_true", help="restart daemons even if nothing changed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("controller", help="set up the controller (run first)")
    n = sub.add_parser("node", help="set up a GPU node")
    n.add_argument("name", help="node name from [[nodes]], e.g. train-01")
    n.add_argument("--munge-key", type=Path, help="munge key copied from the controller")
    sub.add_parser("apply", help="push cluster.toml changes on the controller")
    sub.add_parser("accounts", help="reconcile QOS, accounts and users only")
    sub.add_parser("verify", help="run a test job on every node")
    r = sub.add_parser("render", help="write generated files to a directory")
    r.add_argument("--out", type=Path, default=Path("out"))
    sub.add_parser("validate", help="check cluster.toml")
    sub.add_parser("steps", help="list steps per command")
    return ap.parse_args(argv)


def _load(args) -> team7config.Config:
    path = args.config or (args.cluster_dir / "cluster.toml")
    if not path.exists():
        example = args.cluster_dir / "cluster.example.toml"
        raise InstallError(f"{path} not found. Start from the example: cp {example} {path}")
    try:
        return team7config.load(path)
    except team7config.ConfigError as ex:
        raise InstallError(str(ex)) from None


def _select(plan: list[str], only: str | None, skip: str | None) -> list[str]:
    def names(s):
        out = [x.strip() for x in s.split(",") if x.strip()]
        bad = [x for x in out if x not in STEPS]
        if bad:
            raise InstallError(f"unknown step(s): {', '.join(bad)}. See `install.sh steps`.")
        return out

    steps = list(plan)
    if only:
        wanted = names(only)
        steps = [s for s in steps if s in wanted]
        missing = [s for s in wanted if s not in plan]
        if missing:
            raise InstallError(f"step(s) not part of this command: {', '.join(missing)}")
    if skip:
        drop = set(names(skip))
        steps = [s for s in steps if s not in drop]
    return steps


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.cmd == "steps":
            for cmd, plan in PLANS.items():
                print(f"{cmd:<11} {' '.join(plan)}")
            return 0

        cfg = _load(args)

        if args.cmd == "validate":
            for w in cfg.warnings:
                warn(w)
            ok(f"{cfg.path} is valid")
            return 0

        if args.cmd == "render":
            header(f"Rendering into {args.out}/")
            for rel, content in render.all_files(cfg).items():
                p = args.out / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                info(str(p))
            for w in cfg.warnings:
                warn(w)
            ok("nothing was installed")
            return 0

        role, node_entry = "none", None
        if args.cmd in ("controller", "apply", "accounts", "verify"):
            role = "controller"
        elif args.cmd == "node":
            role = "node"
            try:
                node_entry = cfg.node(args.name)
            except KeyError:
                names = ", ".join(n["name"] for n in cfg["nodes"])
                raise InstallError(f"no node {args.name!r} in {cfg.path}; nodes are: {names}") from None

        ctx = Ctx(
            cfg=cfg,
            cluster_dir=args.cluster_dir.resolve(),
            role=role,
            node=node_entry,
            dry_run=args.dry_run,
            munge_key=getattr(args, "munge_key", None),
            force=args.force,
        )

        if args.cmd == "verify":
            return verify.verify_cluster(ctx)

        plan = _select(PLANS[args.cmd], args.only, args.skip)
        print(f"Team 7 installer: {args.cmd} {ctx.hostname}  ({cfg.path})")
        print("steps: " + " ".join(plan) + ("   [dry-run]" if args.dry_run else ""))
        for name in plan:
            STEPS[name](ctx)
        header("Done")
        if ctx.changed:
            info(f"{len(ctx.changed)} file(s) changed")
        return 0
    except InstallError as ex:
        print(f"\nERROR: {ex}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to continue", file=sys.stderr)
        return 130
