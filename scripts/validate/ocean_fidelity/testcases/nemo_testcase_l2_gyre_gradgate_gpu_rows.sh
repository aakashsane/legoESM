#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; printf "REFUSE gyre-gradgate-gpu: command failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2' ERR

readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly EVIDENCE_DIR=${1:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round7}
readonly GATE=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradient_gate.py
readonly MODEL_PATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src

refuse() {
  printf 'REFUSE gyre-gradgate-gpu: %s\n' "$1" >&2
  exit "${2:-64}"
}

[[ -x "$PYTHON" ]] || refuse "missing Python environment: $PYTHON"
[[ -f "$GATE" ]] || refuse "run from the legoESM repository root; missing $GATE"
[[ -d "$EVIDENCE_DIR" ]] || refuse "evidence directory does not exist: $EVIDENCE_DIR"
[[ -z "$(git status --porcelain)" ]] || refuse "producer worktree is not clean"
readonly COMMIT=$(git rev-parse HEAD)

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env -u XLA_PYTHON_CLIENT_ALLOCATOR \
  "$PYTHON" -c 'import jax; devices=jax.devices(); assert jax.default_backend() == "gpu", jax.default_backend(); assert len(devices) == 1, devices; print(devices[0])'
nvidia-smi -i 1 --query-gpu=index,memory.used --format=csv,noheader

ROW_FAILURES=0

run_row() {
  local output=$1
  local mode=$2
  local steps=$3
  local schedule=$4
  local rc=0
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  if CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$MODEL_PATH" \
      env -u XLA_PYTHON_CLIENT_ALLOCATOR \
      "$PYTHON" "$GATE" "$mode" --steps "$steps" --schedule "$schedule" \
        --transcendentals native \
        --expect-commit "$COMMIT" --output "$output"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ! -f "$output" ]]; then
    printf 'REFUSE gyre-gradgate-gpu: row wrote no JSON (exit %s): %s\n' \
      "$rc" "$*" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if ! "$PYTHON" -c '
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text())
commit, mode, steps, schedule = sys.argv[2:]
expected = "GPU, XLA transcendentals — not the certified bit-exact policy"
assert row["measurement_policy_label"] == expected, row
assert "transcendentals=\x27native\x27" in row["precision_policy"], row
assert row["provenance"] == {
    "commit": commit, "dirty": False, "verified_after_measurement": True}, row
assert row["case"] == "GYRE-zco", row
assert row["backend"] == "gpu", row
assert row["physical_gpu_index"] == 1, row
assert row["cuda_visible_devices"] == "1", row
assert row["steps"] == int(steps), row
assert row["schedule"] == schedule, row
if mode == "resources":
    assert row["resource_schema"] == "gyre-gradgate-resource-v2-compiled-reverse", row
    assert row["reverse_transform"] == "jax.jit(jax.value_and_grad(scalar_line))", row
    assert row["warm_reverse_timer_span"].startswith("blocked execution"), row
    assert row["xla_python_client_preallocate"].lower() == "false", row
    assert row["xla_python_client_allocator"] is None, row
else:
    assert row["schedule_scope"].startswith("Decision-49 per-step"), row
' "$output" "$COMMIT" "$mode" "$steps" "$schedule"; then
    printf 'REFUSE gyre-gradgate-gpu: invalid row provenance: %s\n' "$output" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if ((rc != 0)); then
    printf 'REFUSE gyre-gradgate-gpu: row exited %s after writing JSON: %s\n' \
      "$rc" "$output" >&2
    ((ROW_FAILURES += 1))
  fi
}

run_suite() {
  local output=$1
  local plant=${2:-}
  shift 2 || true
  local rc=0
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  args=(gpu-resource-suite --expect-commit "$COMMIT" --output "$output")
  for row in "$@"; do
    args+=(--input-row "$row")
  done
  [[ -z "$plant" ]] || args+=(--plant "$plant")
  if CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$MODEL_PATH" \
      env -u XLA_PYTHON_CLIENT_ALLOCATOR \
      "$PYTHON" "$GATE" "${args[@]}"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ! -f "$output" ]]; then
    printf 'REFUSE gyre-gradgate-gpu: suite wrote no JSON (exit %s): %s\n' \
      "$rc" "$output" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if [[ -z "$plant" && $rc -ne 0 ]]; then
    printf 'REFUSE gyre-gradgate-gpu: resource suite off-bar: %s\n' "$output" >&2
    ((ROW_FAILURES += 1))
  elif [[ -n "$plant" && $rc -eq 0 ]]; then
    printf 'REFUSE gyre-gradgate-gpu: ordering plant did not flip: %s\n' "$output" >&2
    ((ROW_FAILURES += 1))
  fi
}

for steps in 1 6; do
  run_row "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json" \
    agreement "$steps" per_step
done

for steps in 1 6; do
  for schedule in none uniform binomial; do
    run_row "$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json" \
      resources "$steps" "$schedule"
  done
  rows=(
    "$EVIDENCE_DIR/gpu1_resources_none_${steps}step.json"
    "$EVIDENCE_DIR/gpu1_resources_uniform_${steps}step.json"
    "$EVIDENCE_DIR/gpu1_resources_binomial_${steps}step.json"
  )
  run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step.json" "" "${rows[@]}"
  run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step_plant.json" \
    gpu_ordering "${rows[@]}"
done

((ROW_FAILURES == 0)) || refuse "$ROW_FAILURES measurement rows failed" 1

printf 'GYRE_GRADGATE_GPU_ROWS_COMPLETE commit=%s evidence=%s\n' \
  "$COMMIT" "$EVIDENCE_DIR"
