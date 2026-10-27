#!/usr/bin/env bash
# Team 7 cluster installer. Runs the stdlib-only Python installer with the
# system python3 (3.12 on Ubuntu 24.04). See README.md.
#
#   sudo ./install.sh controller
#   sudo ./install.sh node train-01 --munge-key /tmp/munge.key
#   sudo ./install.sh verify
#   ./install.sh render            # no root needed, installs nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${TEAM7_PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "install.sh: python3 not found" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "install.sh: python3 >= 3.11 required (tomllib); found $("$PY" -V 2>&1)" >&2
  exit 1
fi

export PYTHONPATH="$HERE/installer:$HERE/team7config${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m team7_installer --cluster-dir "$HERE" "$@"
