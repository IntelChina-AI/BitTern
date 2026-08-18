#!/usr/bin/env bash
# Export the selected CAT-Q checkpoint as a packed ternary GGUF model.
#
# The GGUF is written directly from the checkpoint; no llama.cpp checkout and no
# intermediate model are involved.  See deployment/README.md for how to run the
# result.
set -euo pipefail

source ./task_list.conf
source ./scripts/gpu_lock.sh

config_path="${result_dir}/config.yaml"
checkpoint_path="${result_dir}/parameters.pth"
export_path="${result_dir}/export-gguf"
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
  --export_gguf_path "${export_path}" \
  --gguf_float_type f16 \
  --checkpoint "${checkpoint_path}"
