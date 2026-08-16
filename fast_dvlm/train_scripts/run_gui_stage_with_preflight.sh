#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR to a unique experiment directory}"
preflight_dir="${output_dir}-preflight-zero2"
preflight_log="${preflight_dir}/launcher.log"
selection_file="${output_dir}-preflight-selection.json"

if [[ -f "${output_dir}/training-audit.json" ]]; then
  echo "Training already completed: ${output_dir}"
  exit 0
fi

if [[ -f "${selection_file}" ]]; then
  zero_stage="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["zero_stage"])' "${selection_file}")"
  echo "Reusing completed preflight selection: ZeRO-${zero_stage}"
else
  mkdir -p "${preflight_dir}"
  if [[ -f "${preflight_dir}/training-audit.json" ]]; then
    smoke_status=0
  else
    set +e
    OUTPUT_DIR="${preflight_dir}" ZERO_STAGE=2 RECIPE_SMOKE=1 \
      bash "${script_dir}/run_gui_stage.sh" 2>&1 | tee "${preflight_log}"
    smoke_status="${PIPESTATUS[0]}"
    set -e
  fi

  zero_stage=2
  baseline_dir="${preflight_dir}"
  if (( smoke_status != 0 )); then
    if grep -Eqi 'out of memory|CUDA OOM|CUBLAS_STATUS_ALLOC_FAILED' "${preflight_log}"; then
      zero_stage=3
      echo "ZeRO-2 smoke OOM; selecting ZeRO-3 no-offload"
    else
      echo "ZeRO-2 smoke failed for a non-OOM reason; not changing the recipe" >&2
      exit "${smoke_status}"
    fi
  else
    peak="$(python3 - "${preflight_dir}/training-audit.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(float(value["peak_gpu_reserved_gib"] or 0.0))
PY
)"
    if python3 - "${peak}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 72.0 else 1)
PY
    then
      zero_stage=3
      echo "ZeRO-2 smoke peak ${peak} GiB exceeds 72 GiB; selecting ZeRO-3 no-offload"
    else
      echo "ZeRO-2 smoke peak ${peak} GiB is within the 72 GiB limit"
    fi
  fi

  if [[ "${zero_stage}" == 3 ]]; then
    baseline_dir="${output_dir}-preflight-zero3"
    mkdir -p "${baseline_dir}"
    if [[ ! -f "${baseline_dir}/training-audit.json" ]]; then
      OUTPUT_DIR="${baseline_dir}" ZERO_STAGE=3 RECIPE_SMOKE=1 RECIPE_SMOKE_STEPS=2 \
        bash "${script_dir}/run_gui_stage.sh"
    fi
  fi

  resume_dir="${output_dir}-preflight-zero${zero_stage}-resume-schedule-v2"
  mkdir -p "${resume_dir}"
  resume_consistent=0
  if [[ -f "${resume_dir}/resume-consistency.json" ]]; then
    if python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("accepted") else 1)' \
      "${resume_dir}/resume-consistency.json"
    then
      resume_consistent=1
    fi
  fi
  if (( ! resume_consistent )); then
    # Build the exact same two-step scheduler as the uninterrupted control,
    # but stop after step one. Changing max_steps to one changes warmup/cosine
    # state and makes the resume comparison invalid.
    if [[ ! -d "${resume_dir}/checkpoint-1" ]]; then
      OUTPUT_DIR="${resume_dir}" ZERO_STAGE="${zero_stage}" RECIPE_SMOKE=1 \
        RECIPE_SMOKE_STEPS=2 RECIPE_SMOKE_SAVE_STEPS=1 \
        RECIPE_SMOKE_STOP_AFTER_STEPS=1 \
        bash "${script_dir}/run_gui_stage.sh"
    fi
    if [[ ! -d "${resume_dir}/checkpoint-2" ]]; then
      OUTPUT_DIR="${resume_dir}" ZERO_STAGE="${zero_stage}" RECIPE_SMOKE=1 \
        RECIPE_SMOKE_STEPS=2 RECIPE_SMOKE_SAVE_STEPS=1 \
        bash "${script_dir}/run_gui_stage.sh"
    fi
    python3 - "${baseline_dir}" "${resume_dir}" <<'PY'
import json
import sys
from pathlib import Path
from fast_dvlm.gui_finetune.training import compare_saved_model_weights

result = compare_saved_model_weights(Path(sys.argv[1]), Path(sys.argv[2]))
Path(sys.argv[2], "resume-consistency.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not result["accepted"]:
    raise SystemExit("2-step uninterrupted and 1+resume weights exceed numeric tolerances")
print(json.dumps(result, sort_keys=True))
PY
  fi
  python3 - "${selection_file}" "${zero_stage}" "${baseline_dir}" "${resume_dir}" <<'PY'
import json
import os
import sys
from pathlib import Path
value = {
    "schema_version": 1,
    "zero_stage": int(sys.argv[2]),
    "baseline_dir": str(Path(sys.argv[3]).resolve()),
    "resume_dir": str(Path(sys.argv[4]).resolve()),
}
path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
fi
OUTPUT_DIR="${output_dir}" ZERO_STAGE="${zero_stage}" RECIPE_SMOKE=0 \
  exec bash "${script_dir}/run_gui_stage.sh"
