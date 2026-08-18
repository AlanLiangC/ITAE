#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUV_ENV="${SUV_ENV:-suv-navsim1}"

cd "${REPO_ROOT}"
exec conda run --no-capture-output -n "${SUV_ENV}" \
  python -u -m tools.suv.evaluate_navsim_v1 evaluate "$@"
