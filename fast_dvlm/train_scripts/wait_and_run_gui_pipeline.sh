#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
work_root="${WORK_ROOT:?Set WORK_ROOT to the unique experiment directory}"
env_dir="${FAST_DVLM_ENV:?Set FAST_DVLM_ENV to the prepared Python environment}"
poll_seconds="${GPU_WAIT_POLL_SECONDS:-60}"
minimum_disk_gib="${MINIMUM_DISK_GIB:-300}"
minimum_gpu_mib="${MINIMUM_GPU_FREE_MIB:-71680}"

if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_WAIT_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "${env_dir}/bin/python3" || ! -x "${env_dir}/bin/deepspeed" ]]; then
  echo "FAST_DVLM_ENV is missing python3 or deepspeed: ${env_dir}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to train from a dirty source tree: ${repo_root}" >&2
  exit 2
fi

mkdir -p "${work_root}"
exec 9>"${work_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another GUI pipeline waiter already owns ${work_root}/pipeline.lock" >&2
  exit 2
fi

export PATH="${env_dir}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

revision="$(git -C "${repo_root}" rev-parse HEAD)"
printf 'waiter_started=%s revision=%s work_root=%s\n' \
  "$(date --iso-8601=seconds)" "${revision}" "${work_root}"

while true; do
  disk_free_gib="$(df -Pk "${work_root}" | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
  mapfile -t gpu_free < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -2
  )
  ready=1
  reason=()
  if (( disk_free_gib < minimum_disk_gib )); then
    ready=0
    reason+=("disk=${disk_free_gib}GiB")
  fi
  if [[ "${#gpu_free[@]}" -ne 2 ]]; then
    ready=0
    reason+=("gpus=${#gpu_free[@]}")
  else
    for index in 0 1; do
      if (( gpu_free[index] < minimum_gpu_mib )); then
        ready=0
        reason+=("gpu${index}=${gpu_free[index]}MiB")
      fi
    done
  fi

  if (( ready )); then
    printf 'resources_ready=%s disk_free_gib=%s gpu_free_mib=%s,%s\n' \
      "$(date --iso-8601=seconds)" "${disk_free_gib}" "${gpu_free[0]}" "${gpu_free[1]}"
    set +e
    RESUME_PIPELINE=1 bash "${script_dir}/run_gui_pipeline.sh"
    status=$?
    set -e
    if (( status == 75 )); then
      echo "A stage lost the 70-GiB GPU reservation race; returning to the safe wait loop"
    else
      exit "${status}"
    fi
  else
    printf 'resources_waiting=%s %s\n' "$(date --iso-8601=seconds)" "${reason[*]}"
  fi
  sleep "${poll_seconds}"
done
