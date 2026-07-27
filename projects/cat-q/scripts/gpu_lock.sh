#!/usr/bin/env bash

CATQ_GPU_LOCK_DIR="${CATQ_GPU_LOCK_DIR:-${TMPDIR:-/tmp}/catq-gpu-locks-${USER:-user}}"
CATQ_ACQUIRED_GPUS=""

catq_gpu_memory() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null
}

catq_release_gpus() {
  local gpu
  for gpu in ${CATQ_ACQUIRED_GPUS//,/ }; do
    rm -f "${CATQ_GPU_LOCK_DIR}/gpu_${gpu}.lock"
  done
  CATQ_ACQUIRED_GPUS=""
}

catq_try_lock_gpu() {
  local gpu="$1"
  local threshold="$2"
  local lockfile="${CATQ_GPU_LOCK_DIR}/gpu_${gpu}.lock"
  local guard="${lockfile}.guard"
  local used total

  if ! mkdir "${guard}" 2>/dev/null; then
    return 1
  fi

  read -r used total < <(catq_gpu_memory | sed -n "$((gpu + 1))p" | awk -F',' '{print $1, $2}')
  if [[ ! -e "${lockfile}" ]] && awk -v used="${used:-0}" -v total="${total:-0}" -v threshold="${threshold}" \
    'BEGIN { exit !(total > 0 && used / total < threshold) }'; then
    printf '%s\n' "${BASHPID}" > "${lockfile}"
    rmdir "${guard}"
    return 0
  fi

  rmdir "${guard}"
  return 1
}

catq_acquire_gpus() {
  local needed="$1"
  local threshold="$2"
  local wait_mode="$3"
  local wait_interval="$4"
  local total_gpus gpu
  local -a acquired

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for automatic GPU allocation." >&2
    return 1
  fi

  mkdir -p "${CATQ_GPU_LOCK_DIR}"
  while true; do
    acquired=()
    total_gpus="$(catq_gpu_memory | wc -l | tr -d ' ')"
    for ((gpu = 0; gpu < total_gpus; gpu++)); do
      if catq_try_lock_gpu "${gpu}" "${threshold}"; then
        acquired+=("${gpu}")
        if [[ "${#acquired[@]}" -eq "${needed}" ]]; then
          CATQ_ACQUIRED_GPUS="$(IFS=,; echo "${acquired[*]}")"
          export CUDA_VISIBLE_DEVICES="${CATQ_ACQUIRED_GPUS}"
          echo "Acquired GPUs: ${CATQ_ACQUIRED_GPUS}"
          return 0
        fi
      fi
    done

    CATQ_ACQUIRED_GPUS="$(IFS=,; echo "${acquired[*]}")"
    catq_release_gpus
    if [[ "${wait_mode}" != "true" ]]; then
      echo "Could not acquire ${needed} GPU(s)." >&2
      return 1
    fi
    echo "Waiting for ${needed} GPU(s); retrying in ${wait_interval}s."
    sleep "${wait_interval}"
  done
}
