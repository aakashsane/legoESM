#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: swg_dg_rankcheck.sh --nemo-cfg DIR --output DIR [--mpiexec CMD] [--python PYTHON] [--execute]

Default mode is a read-only preflight. --execute runs the committed Maddison
deck for ten steps on one and 32 ranks, then compares every restart variable
word for word. No path defaults to a developer machine.
EOF
}

nemo_cfg=
output=
mpiexec=mpirun
python=python3
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nemo-cfg) nemo_cfg=${2-}; shift ;;
    --output) output=${2-}; shift ;;
    --mpiexec) mpiexec=${2-}; shift ;;
    --python) python=${2-}; shift ;;
    --execute) execute=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'REFUSE_USAGE: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
  shift
done
[[ -n "$nemo_cfg" && -n "$output" ]] || {
  printf 'REFUSE_USAGE: --nemo-cfg and --output are required\n' >&2; exit 64; }
[[ -x "$nemo_cfg/BLD/bin/nemo.exe" ]] || {
  printf 'REFUSE_NEMO_CFG: missing built executable under %s\n' "$nemo_cfg" >&2; exit 66; }
[[ ! -e "$output" ]] || { printf 'REFUSE_OUTPUT_EXISTS: %s\n' "$output" >&2; exit 69; }

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: tool is outside a Git checkout\n' >&2; exit 66; }
runner=$here/swg_md_run.sh
checker=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round6.py
printf 'PREFLIGHT rank-control=1-versus-32 steps=10 deck=maddison\n'
if [[ "$execute" -ne 1 ]]; then
  printf 'WOULD_RUN: %s ... np1 16 50 10 1 10\n' "$runner"
  printf 'WOULD_RUN: %s ... np32 16 50 10 32 10\n' "$runner"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$output"
"$runner" --nemo-cfg "$nemo_cfg" --output-root "$output" \
  --mpiexec "$mpiexec" --execute np1 16 50 10 1 10
"$runner" --nemo-cfg "$nemo_cfg" --output-root "$output" \
  --mpiexec "$mpiexec" --execute np32 16 50 10 32 10
"$python" "$checker" rank-check --root "$output" \
  --json "$output/rank_identity.json"
