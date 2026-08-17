#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
eval_dir="${repo_root}/fast_dvlm/eval"
export PYTHONPATH="${repo_root}/third_party/sglang/python:${repo_root}/third_party:${repo_root}:${PYTHONPATH:-}"

work_root="${WORK_ROOT:?Set WORK_ROOT to a new, unique experiment directory}"
if [[ -e "${work_root}" ]]; then
  if [[ "${RESUME_PIPELINE:-0}" != 1 ]]; then
    echo "Experiment directory exists; set RESUME_PIPELINE=1 to resume: ${work_root}" >&2
    exit 2
  fi
  echo "Resuming experiment directory: ${work_root}"
else
  mkdir -p "${work_root}"/{data,training,validation,final}
fi
mkdir -p "${work_root}"/{data,training,validation,final}

planner_source="${PLANNER_SOURCE:-/home/ma-user/work/LLaDA-o/data/unigui-openmobile-planner-v2-content-v4}"
planner_images="${PLANNER_IMAGES:-/home/ma-user/work/LLaDA-Agent/data/Uni-GUI-OpenMobile}"
mind2web_train="${MIND2WEB_TRAIN:-/home/ma-user/work/LLaDA-o/data/train_ocr/mind2web}"
mobile_train="${MOBILE_TRAIN:-/home/ma-user/work/LLaDA-o/data/residual-grounding/mobile/train}"
mind2web_validation="${MIND2WEB_VALIDATION:-/home/ma-user/work/LLaDA-o/data/bench_ocr_validation}"
mind2web_validation_key="${MIND2WEB_VALIDATION_KEY:-mind2web_validation}"
mobile_validation="${MOBILE_VALIDATION:-/home/ma-user/work/LLaDA-o/data/residual-grounding/mobile/benchmark}"
mobile_validation_key="${MOBILE_VALIDATION_KEY:-mobile_validation}"
mind2web_test="${MIND2WEB_TEST:?Set MIND2WEB_TEST to the fixed OCR-aligned test-100 root}"
mind2web_test_key="${MIND2WEB_TEST_KEY:?Set MIND2WEB_TEST_KEY to its manifest key}"
test_ids_sha256="${MIND2WEB_TEST_IDS_SHA256:-00a91fdb996afae3bd14af096eecf3fbf95535cb482582d4bc203476f212a689}"

prepare_data() {
  local output="$1"
  shift
  if [[ -f "${output}/train.json" && -f "${output}/audit.json" ]]; then
    python3 - "${output}/train.json" <<'PY'
import sys
from pathlib import Path
from fast_dvlm.gui_finetune.data import audit_converted_training_file
print(audit_converted_training_file(Path(sys.argv[1])))
PY
    echo "Reusing authenticated converted data: ${output}"
    return
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing incomplete converted-data directory: ${output}" >&2
    exit 2
  fi
  "$@"
}

prepare_data "${work_root}/data/planner" \
  python3 "${repo_root}/fast_dvlm/data/prepare_gui_data.py" planner \
    --source-dir "${planner_source}" --image-root "${planner_images}" \
    --output-dir "${work_root}/data/planner"
prepare_data "${work_root}/data/grounder" \
  python3 "${repo_root}/fast_dvlm/data/prepare_gui_data.py" grounder \
    --mind2web-dir "${mind2web_train}" --mobile-dir "${mobile_train}" \
    --output-dir "${work_root}/data/grounder" \
    --mind2web-validation-root "${mind2web_validation}" \
    --mind2web-validation-key "${mind2web_validation_key}" \
    --mobile-validation-root "${mobile_validation}" \
    --mobile-validation-key "${mobile_validation_key}" \
    --mind2web-test-root "${mind2web_test}" \
    --mind2web-test-key "${mind2web_test_key}"

planner_output="${work_root}/training/planner"
STAGE=planner \
DATASET_PATH="${work_root}/data/planner/train.json" \
SOURCE_AUDIT="${work_root}/data/planner/audit.json" \
OUTPUT_DIR="${planner_output}" \
MODEL_PATH="Efficient-Large-Model/Fast_dVLM_3B" \
TOKENIZER_NAME="Efficient-Large-Model/Fast_dVLM_3B" \
bash "${script_dir}/run_gui_stage_with_preflight.sh"

run_eval() {
  local task="$1" model="$2" adapter="$3" dataset_or_root="$4" benchmark="$5" algorithm="$6" output="$7" expected_hash="${8:-}"
  if [[ "${task}" == planner ]]; then
    TASK=planner MODEL_PATH="${model}" PROCESSOR_PATH="${planner_output}" \
      DATASET="${dataset_or_root}" ALGORITHM="${algorithm}" OUTPUT_DIR="${output}" \
      EXPECTED_SAMPLE_IDS_SHA256="${expected_hash}" \
      bash "${eval_dir}/run_gui_eval_2gpu.sh"
  else
    TASK=grounder MODEL_PATH="${model}" PROCESSOR_PATH="${planner_output}" \
      ADAPTER_PATH="${adapter}" BENCHMARK_ROOT="${dataset_or_root}" \
      BENCHMARK="${benchmark}" ALGORITHM="${algorithm}" OUTPUT_DIR="${output}" \
      EXPECTED_SAMPLE_IDS_SHA256="${expected_hash}" \
      bash "${eval_dir}/run_gui_eval_2gpu.sh"
  fi
}

planner_validation_ids_sha256="$(python3 - "${work_root}/data/planner/audit.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["validation_selection"]["sample_ids_sha256"])
PY
)"
planner_candidates=()
for step in 813 1626; do
  for algorithm in mdm spec; do
    suffix="-${algorithm}"
    if [[ "${algorithm}" == mdm ]]; then
      # Keep the original MDM path stable so interrupted experiments can
      # authenticate and reuse their completed validation output.
      suffix=""
    fi
    predictions="${work_root}/validation/planner-${step}${suffix}"
    score="${predictions}/score.json"
    run_eval planner "${planner_output}/checkpoint-${step}" "" \
      "${work_root}/data/planner/validation-100.json" "" "${algorithm}" "${predictions}" \
      "${planner_validation_ids_sha256}"
    python3 "${eval_dir}/score_gui.py" --task planner --predictions-dir "${predictions}" \
      --expected-samples 100 --step "${step}" --algorithm "${algorithm}" --output "${score}"
    planner_candidates+=(--candidate "${score}")
  done
done
planner_selection="${work_root}/validation/planner-selection.json"
python3 "${eval_dir}/select_checkpoint.py" --stage planner \
  "${planner_candidates[@]}" --output "${planner_selection}"
planner_step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["step"])' "${planner_selection}")"
planner_backbone="${planner_output}/checkpoint-${planner_step}"

grounder_output="${work_root}/training/grounder"
STAGE=grounder \
DATASET_PATH="${work_root}/data/grounder/train.json" \
SOURCE_AUDIT="${work_root}/data/grounder/audit.json" \
OUTPUT_DIR="${grounder_output}" \
MODEL_PATH="${planner_backbone}" TOKENIZER_NAME="${planner_output}" \
bash "${script_dir}/run_gui_stage_with_preflight.sh"

grounder_candidates=()
for step in 1033 2066 3099; do
  adapter="${grounder_output}/checkpoint-${step}"
  base="${work_root}/validation/grounder-${step}-mdm"
  run_eval grounder "${planner_backbone}" "${adapter}" "${mind2web_validation}" \
    "${mind2web_validation_key}" mdm "${base}-mind2web"
  run_eval grounder "${planner_backbone}" "${adapter}" "${mobile_validation}" \
    "${mobile_validation_key}" mdm "${base}-mobile"
  python3 "${eval_dir}/score_gui.py" --task grounder --predictions-dir "${base}-mind2web" \
    --expected-samples 100 --step "${step}" --algorithm mdm --output "${base}-mind2web-score.json"
  python3 "${eval_dir}/score_gui.py" --task grounder --predictions-dir "${base}-mobile" \
    --expected-samples 100 --step "${step}" --algorithm mdm --output "${base}-mobile-score.json"
  combined="${base}-combined.json"
  python3 "${eval_dir}/combine_grounder_validation.py" --step "${step}" --algorithm mdm \
    --mind2web "${base}-mind2web-score.json" --mobile "${base}-mobile-score.json" \
    --output "${combined}"
  grounder_candidates+=(--candidate "${combined}")
done
grounder_selection="${work_root}/validation/grounder-selection.json"
python3 "${eval_dir}/select_checkpoint.py" --stage grounder \
  "${grounder_candidates[@]}" --output "${grounder_selection}"
grounder_step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["step"])' "${grounder_selection}")"
grounder_adapter="${grounder_output}/checkpoint-${grounder_step}"
mdm_candidate="${work_root}/validation/grounder-${grounder_step}-mdm-combined.json"

spec_base="${work_root}/validation/grounder-${grounder_step}-spec"
run_eval grounder "${planner_backbone}" "${grounder_adapter}" "${mind2web_validation}" \
  "${mind2web_validation_key}" spec "${spec_base}-mind2web"
run_eval grounder "${planner_backbone}" "${grounder_adapter}" "${mobile_validation}" \
  "${mobile_validation_key}" spec "${spec_base}-mobile"
python3 "${eval_dir}/score_gui.py" --task grounder --predictions-dir "${spec_base}-mind2web" \
  --expected-samples 100 --step "${grounder_step}" --algorithm spec --output "${spec_base}-mind2web-score.json"
python3 "${eval_dir}/score_gui.py" --task grounder --predictions-dir "${spec_base}-mobile" \
  --expected-samples 100 --step "${grounder_step}" --algorithm spec --output "${spec_base}-mobile-score.json"
spec_candidate="${spec_base}-combined.json"
python3 "${eval_dir}/combine_grounder_validation.py" --step "${grounder_step}" --algorithm spec \
  --mind2web "${spec_base}-mind2web-score.json" --mobile "${spec_base}-mobile-score.json" \
  --output "${spec_candidate}"
inference_selection="${work_root}/validation/inference-selection.json"
python3 "${eval_dir}/select_inference.py" --candidate "${mdm_candidate}" \
  --candidate "${spec_candidate}" --output "${inference_selection}"
algorithm="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["algorithm"])' "${inference_selection}")"

# Prove on validation data that Planner and Grounder share one resident model
# object and that disabling the pinned adapter restores the Planner output.
shared_runtime_audit="${work_root}/validation/shared-runtime-audit.json"
if [[ -f "${shared_runtime_audit}" ]]; then
  python3 - "${shared_runtime_audit}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if not value.get("one_backbone") or not value.get("planner_output_restored"):
    raise SystemExit("existing shared runtime audit is invalid")
print("Reusing authenticated shared runtime audit")
PY
else
  mapfile -t shared_runtime_gpu_free < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -2
  )
  if [[ "${#shared_runtime_gpu_free[@]}" -ne 2 ]]; then
    echo "shared runtime audit requires exactly two visible GPUs" >&2
    exit 75
  fi
  for mib in "${shared_runtime_gpu_free[@]}"; do
    if (( mib < 71680 )); then
      echo "shared runtime audit requires both GPUs to remain at least 70 GiB free" >&2
      exit 75
    fi
  done
  shared_runtime_image="$(python3 - "${mind2web_validation}" "${mind2web_validation_key}" <<'PY'
import sys
from pathlib import Path
from fast_dvlm.gui_finetune.data import load_benchmark_rows
root = Path(sys.argv[1]).resolve()
rows, _ = load_benchmark_rows(root, sys.argv[2])
print((root / rows[0]["image"]).resolve())
PY
)"
  CUDA_VISIBLE_DEVICES=0 python3 "${eval_dir}/verify_shared_runtime.py" \
    --model-path "${planner_backbone}" --adapter-path "${grounder_adapter}" \
    --processor-path "${planner_output}" --image "${shared_runtime_image}" \
    --algorithm "${algorithm}" --output "${shared_runtime_audit}"
fi

# The held-out test set is touched exactly once, after all checkpoint and
# inference choices are frozen on validation.
final_predictions="${work_root}/final/mind2web-ocr-test100"
run_eval grounder "${planner_backbone}" "${grounder_adapter}" "${mind2web_test}" \
  "${mind2web_test_key}" "${algorithm}" "${final_predictions}" "${test_ids_sha256}"
final_score="${work_root}/final/mind2web-ocr-test100-score.json"
python3 "${eval_dir}/score_gui.py" --task grounder --predictions-dir "${final_predictions}" \
  --expected-samples 100 --step "${grounder_step}" --algorithm "${algorithm}" \
  --output "${final_score}"
python3 "${eval_dir}/report_gui.py" --planner-selection "${planner_selection}" \
  --grounder-selection "${grounder_selection}" --inference-selection "${inference_selection}" \
  --final-score "${final_score}" --model-dir "${planner_backbone}" \
  --adapter-dir "${grounder_adapter}" \
  --gpu-memory-audit "${final_predictions}/gpu-memory-audit.json" \
  --shared-runtime-audit "${shared_runtime_audit}" \
  --output "${work_root}/final/comparison.json"
