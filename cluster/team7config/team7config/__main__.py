"""Command line access to the cluster TOML.

    python3 -m team7config validate [--config PATH]
    python3 -m team7config get slurm.requeue_exit [--config PATH]
    python3 -m team7config show [--config PATH]

`get` prints scalars as-is, lists one item per line, tables as JSON.
Exit code 0 on success, 1 on validation errors, 2 on a missing key.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import ConfigError, load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="team7-config")
    ap.add_argument("--config", help="TOML path (default: $TEAM7_CONFIG or /etc/team7/cluster.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check the file and print warnings")
    g = sub.add_parser("get", help="print one value by dotted key")
    g.add_argument("key")
    sub.add_parser("show", help="print the full config with defaults filled in, as JSON")
    args = ap.parse_args(argv)

    try:
        cfg = load(args.config)
    except ConfigError as ex:
        print(ex, file=sys.stderr)
        return 1

    if args.cmd == "validate":
        for w in cfg.warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(f"OK: {cfg.path}")
        return 0
    if args.cmd == "show":
        json.dump(cfg.data, sys.stdout, indent=2)
        print()
        return 0
    sentinel = object()
    value = cfg.get(args.key, sentinel)
    if value is sentinel:
        print(f"no such key: {args.key}", file=sys.stderr)
        return 2
    if isinstance(value, bool):
        print("true" if value else "false")
    elif isinstance(value, list):
        for item in value:
            print(json.dumps(item) if isinstance(item, (dict, list)) else item)
    elif isinstance(value, dict):
        print(json.dumps(value))
    elif value is None:
        print("")
    else:
        print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
