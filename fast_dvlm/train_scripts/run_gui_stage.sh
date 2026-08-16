#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${repo_root}/third_party:${repo_root}:${PYTHONPATH:-}"

stage="${STAGE:?Set STAGE=planner or STAGE=grounder}"
dataset_path="${DATASET_PATH:?Set DATASET_PATH to converted train.json}"
source_audit="${SOURCE_AUDIT:?Set SOURCE_AUDIT to the conversion audit.json}"
output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR to a unique experiment directory}"
model_path="${MODEL_PATH:-Efficient-Large-Model/Fast_dVLM_3B}"
tokenizer_name="${TOKENIZER_NAME:-${model_path}}"
zero_stage="${ZERO_STAGE:-2}"

if [[ "${stage}" != planner && "${stage}" != grounder ]]; then
  echo "STAGE must be planner or grounder" >&2
  exit 2
fi
if [[ ! -f "${dataset_path}" || ! -f "${source_audit}" ]]; then
  echo "Converted dataset or source audit is missing" >&2
  exit 2
fi
if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
  echo "OUTPUT_DIR exists and is not a directory: ${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_dir}"

python3 - "${output_dir}" <<'PY'
import shutil
import sys
free = shutil.disk_usage(sys.argv[1]).free
if free < 300 * 1024**3:
    raise SystemExit(f"need at least 300 GiB free, found {free / 1024**3:.1f} GiB")
print(f"disk_free_gib={free / 1024**3:.1f}")
PY

mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -2)
if [[ "${#gpu_free[@]}" -ne 2 ]]; then
  echo "exactly two visible GPUs are required" >&2
  exit 2
fi
for mib in "${gpu_free[@]}"; do
  if (( mib < 71680 )); then
    echo "each GPU needs at least 70 GiB free; observed ${mib} MiB" >&2
    exit 75
  fi
done

if [[ "${zero_stage}" == 2 ]]; then
  ds_config="${repo_root}/fast_dvlm/configs/ds_zero2_no_offload.json"
elif [[ "${zero_stage}" == 3 ]]; then
  ds_config="${repo_root}/fast_dvlm/configs/ds_zero3_no_offload.json"
else
  echo "ZERO_STAGE must be 2 or 3" >&2
  exit 2
fi

if [[ "${stage}" == planner ]]; then
  epochs=2
  learning_rate=1e-6
  save_steps=813
  max_steps=-1
  balance_power=0.25
else
  epochs=3
  learning_rate=1e-5
  save_steps=1033
  max_steps=3099
  balance_power=1.0
fi
recipe_override_args=()
if [[ "${RECIPE_SMOKE:-0}" == 1 ]]; then
  epochs=1
  max_steps="${RECIPE_SMOKE_STEPS:-2}"
  save_steps="${RECIPE_SMOKE_SAVE_STEPS:-${max_steps}}"
  recipe_override_args=(--allow_recipe_override true)
  if [[ -n "${RECIPE_SMOKE_STOP_AFTER_STEPS:-}" ]]; then
    recipe_override_args+=(
      --preflight_stop_after_steps "${RECIPE_SMOKE_STOP_AFTER_STEPS}"
    )
  fi
fi

resume_args=()
latest="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "${latest}" ]]; then
  resume_args=(--resume_from_checkpoint "${latest}")
fi

cd "${repo_root}"
exec deepspeed --num_gpus=2 --master_port="${MASTER_PORT:-29631}" \
  fast_dvlm/train_scripts/train_gui.py \
  --stage "${stage}" \
  --model_name_or_path "${model_path}" \
  --tokenizer_name "${tokenizer_name}" \
  --trust_remote_code true \
  --dataset_path "${dataset_path}" \
  --image_folder / \
  --source_audit "${source_audit}" \
  --output_dir "${output_dir}" \
  "${resume_args[@]}" \
  --num_train_epochs "${epochs}" \
  --max_steps "${max_steps}" \
  --learning_rate "${learning_rate}" \
  --language_learning_rate "${learning_rate}" \
  --connector_learning_rate 2e-6 \
  --vision_learning_rate 1e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --min_lr_ratio 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --weight_decay 0.05 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --balance_power "${balance_power}" \
  --max_pixels 705600 \
  --deepspeed "${ds_config}" \
  --bf16 true \
  --gradient_checkpointing true \
  --seed 42 \
  --data_seed 42 \
  --save_strategy steps \
  --save_steps "${save_steps}" \
  --save_total_limit 4 \
  --logging_steps 1 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --remove_unused_columns false \
  --report_to none \
  --ddp_timeout 72000 \
  --do_train true \
  "${recipe_override_args[@]}"
