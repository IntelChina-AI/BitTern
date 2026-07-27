#!/usr/bin/env bash
set -euo pipefail

source ./task_list.conf
source ./scripts/gpu_lock.sh

config_path="${result_dir}/config.yaml"
checkpoint_path="${result_dir}/parameters.pth"
export_path="${result_dir}/export"
if [[ ! -f "${config_path}" ]]; then
  echo "Config file not found: ${config_path}" >&2
  exit 2
fi
if [[ ! -f "${checkpoint_path}" ]]; then
  echo "Checkpoint not found: ${checkpoint_path}" >&2
  exit 2
fi

mkdir -p "${export_path}"
catq_acquire_gpus 1 "${THRESHOLD}" "${WAIT_MODE}" "${WAIT_INTERVAL}"
trap catq_release_gpus EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python main.py \
  --config "${config_path}" \
  --output_dir "${export_path}" \
  --export_model_path "${export_path}" \
  --checkpoint "${checkpoint_path}"
