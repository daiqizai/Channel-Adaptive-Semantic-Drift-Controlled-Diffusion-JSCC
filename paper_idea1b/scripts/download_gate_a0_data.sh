#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "${ROOT}/paper_idea1b/scripts/parallel_direct_download.py" \
  --config "${ROOT}/paper_idea1b/configs/gate_a0_benchmark_setup.yaml" \
  --dataset all
