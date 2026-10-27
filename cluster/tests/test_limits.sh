#!/usr/bin/env bash
# Wrapper so the test can be run under the name used in week1_plan.md.
#   sudo ./tests/test_limits.sh [--only T07,T08] [--evidence DIR]
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_limits.py" "$@"
