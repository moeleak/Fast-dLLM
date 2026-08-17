#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${repo_root}/third_party/sglang/python:${repo_root}/third_party:${repo_root}:${PYTHONPATH:-}"

task="${TASK:?Set TASK=planner or TASK=grounder}"
model_path="${MODEL_PATH:?Set MODEL_PATH}"
output_dir="${OUTPUT_DIR:?Set a new OUTPUT_DIR}"
algorithm="${ALGORITHM:-mdm}"
if [[ "${task}" != planner && "${task}" != grounder ]]; then
  echo "TASK must be planner or grounder" >&2
  exit 2
fi
write_gpu_audit() {
  python3 - "${output_dir}/gpu-memory.csv" "${output_dir}/gpu-memory-audit.json" <<'PY'
import csv
import json
import sys
from pathlib import Path
peak = {}
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.reader(handle):
        if len(row) != 3:
            continue
        index, used = int(row[1].strip()), int(row[2].strip())
        peak[index] = max(peak.get(index, 0), used)
result = {"schema_version": 1, "peak_memory_used_mib": peak}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
}

if [[ -e "${output_dir}" ]]; then
  if [[ -f "${output_dir}/part-00000.jsonl" && -f "${output_dir}/part-00001.jsonl" &&
        -f "${output_dir}/run-config-00000.json" && -f "${output_dir}/run-config-00001.json" &&
        -f "${output_dir}/gpu-memory.csv" ]]; then
    python3 - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("part-*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
ids = [str(row["sample_id"]) for row in rows]
if len(rows) != 100 or len(set(ids)) != 100:
    raise SystemExit(f"incomplete evaluation output: rows={len(rows)} unique={len(set(ids))}")
if any(row.get("error") for row in rows):
    raise SystemExit("completed evaluation output contains runtime errors")
PY
    write_gpu_audit
    echo "Reusing completed 100-sample evaluation: ${output_dir}"
    exit 0
  fi
  echo "Refusing incomplete evaluation output to avoid rerunning held-out samples: ${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_dir}"

mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -2)
for mib in "${gpu_free[@]}"; do
  if (( mib < 71680 )); then
    echo "each GPU needs at least 70 GiB free; observed ${mib} MiB" >&2
    exit 75
  fi
done

common=(
  --task "${task}"
  --model-path "${model_path}"
  --output-dir "${output_dir}"
  --world-size 2
  --limit 100
  --algorithm "${algorithm}"
  --seed 42
  --fail-fast
)
if [[ -n "${EXPECTED_SAMPLE_IDS_SHA256:-}" ]]; then
  common+=(--expected-sample-ids-sha256 "${EXPECTED_SAMPLE_IDS_SHA256}")
fi
if [[ -n "${PROCESSOR_PATH:-}" ]]; then
  common+=(--processor-path "${PROCESSOR_PATH}")
fi
if [[ "${task}" == planner ]]; then
  common+=(--dataset "${DATASET:?Set DATASET for Planner evaluation}")
else
  common+=(
    --adapter-path "${ADAPTER_PATH:?Set ADAPTER_PATH for Grounder evaluation}"
    --benchmark-root "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}"
    --benchmark "${BENCHMARK:?Set BENCHMARK}"
  )
fi

monitor_log="${output_dir}/gpu-memory.csv"
(
  while true; do
    timestamp="$(date --iso-8601=ns)"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      sed "s/^/${timestamp},/"
    sleep 0.2
  done
) >"${monitor_log}" &
monitor_pid=$!
cleanup() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 python3 "${script_dir}/evaluate_gui.py" "${common[@]}" --rank 0 \
  >"${output_dir}/worker-0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python3 "${script_dir}/evaluate_gui.py" "${common[@]}" --rank 1 \
  >"${output_dir}/worker-1.log" 2>&1 &
pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
cleanup
trap - EXIT INT TERM
if (( status != 0 )); then
  tail -n 100 -- "${output_dir}"/worker-*.log >&2 || true
  exit "${status}"
fi

write_gpu_audit
