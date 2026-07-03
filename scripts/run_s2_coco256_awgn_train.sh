#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/coco outputs/logs

WGET_ARGS=(
  --no-proxy
  -c
  -nv
  --tries=0
  --waitretry=15
  --retry-connrefused
  --read-timeout=60
  --timeout=60
  --retry-on-http-error=502,503,504
)

count_jpgs() {
  find "$1" -type f -name "*.jpg" | wc -l | tr -d " "
}

ensure_coco_split() {
  local split="$1"
  local expected_count="$2"
  local url="http://images.cocodataset.org/zips/${split}.zip"
  local zip_path="data/coco/${split}.zip"
  local split_dir="data/coco/${split}"

  if [[ ! -d "$split_dir" ]]; then
    echo "Downloading ${split}.zip"
    wget "${WGET_ARGS[@]}" -P data/coco "$url"
    echo "Unpacking ${zip_path}"
    unzip -q -n "$zip_path" -d data/coco
  fi

  local count
  count="$(count_jpgs "$split_dir")"
  echo "${split}: ${count}/${expected_count} jpg files"
  if [[ "$count" != "$expected_count" ]]; then
    echo "Unexpected ${split} image count: ${count}, expected ${expected_count}" >&2
    exit 1
  fi
}

echo "Started at $(date -Is)"
python3 -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi || true

ensure_coco_split val2017 5000
ensure_coco_split train2017 118287

python3 scripts/train_deepjscc_highres.py \
  --config configs/s2_deepjscc_coco256_awgn.yaml \
  --device cuda:0 \
  "$@"

echo "Finished at $(date -Is)"
