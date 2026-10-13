#!/usr/bin/env bash
set -Eeuo pipefail

# Operator-only builder.  With no argument this applies and validates the
# patch in a temporary directory; --run creates the dedicated NEMO copy.
do_run=0
[[ $# -eq 0 ]] || [[ $# -eq 1 && $1 == --run ]] || {
  printf 'REFUSE_USAGE: expected [--run]\n' >&2; exit 64; }
[[ $# -eq 0 ]] || do_run=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_NAME=GYRE_OMIP_L2_P3
readonly TARGET_NAME=GYRE_OMIP_L2_P3_DAILY
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/$SOURCE_NAME
readonly TARGET_CFG=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly PATCH=$here/gyre_p3_daily_writer.patch
readonly SOURCE_SHA=11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee
readonly PATCHED_SHA=a3cc1e37ab362f62bc5aba8b3f4ae0a051a5665ea1491c2043220dd5edd41d83

[[ -f $PATCH && -f $SOURCE_CFG/MY_SRC/stprk3.F90 ]] || {
  printf 'REFUSE_MISSING_INPUT: P3 source or daily patch\n' >&2; exit 67; }
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $SOURCE_SHA ]] || {
  printf 'REFUSE_SOURCE_IDENTITY: stprk3.F90 changed\n' >&2; exit 68; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$SOURCE_CFG/MY_SRC/stprk3.F90" "$dry/"
patch -s --fuzz=0 -d "$dry" -p1 <"$PATCH"
[[ $(sha256sum "$dry/stprk3.F90" | awk '{print $1}') == $PATCHED_SHA ]] || {
  printf 'REFUSE_PATCH_IDENTITY: daily patch result changed\n' >&2; exit 68; }
grep -q 'MOD( kstp - nit000, 24 ) == 0' "$dry/stprk3.F90" || {
  printf 'REFUSE_PATCH_PROGRAM: daily cadence missing\n' >&2; exit 68; }
printf 'PREFLIGHT: source=%s target=%s patch=%s\n' "$SOURCE_NAME" "$TARGET_NAME" "$PATCH"
if [[ $do_run -ne 1 ]]; then
  printf 'WOULD_BUILD: %s --run\nPREFLIGHT_OK\n' "$0"
  exit 0
fi

[[ ! -e $TARGET_CFG ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$TARGET_CFG" >&2; exit 69; }
cd "$NEMO_ROOT"
./makenemo -r "$SOURCE_NAME" -n "$TARGET_NAME" -d "OCE" -m conda -j 0
patch -s --fuzz=0 -d "$TARGET_CFG/MY_SRC" -p1 <"$PATCH"
[[ $(sha256sum "$TARGET_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $PATCHED_SHA ]] || {
  printf 'REFUSE_PATCH_IDENTITY: installed daily source changed\n' >&2; exit 68; }
readonly WORK_CFGS=$NEMO_ROOT/cfgs/work_cfgs.txt
[[ -f $WORK_CFGS ]] || {
  printf 'REFUSE_MISSING_INPUT: %s\n' "$WORK_CFGS" >&2; exit 67; }
if ! grep -qE "^${TARGET_NAME}[[:space:]]+OCE([[:space:]]|$)" "$WORK_CFGS"; then
  printf '%s OCE\n' "$TARGET_NAME" >>"$WORK_CFGS"
fi
./makenemo -n "$TARGET_NAME" -m conda -j 8
[[ $(sha256sum "$TARGET_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $PATCHED_SHA ]] || {
  printf 'REFUSE_BUILD_OVERWROTE_PATCH: daily MY_SRC changed\n' >&2; exit 70; }
grep -q 'MOD( kstp - nit000, 24 ) == 0' \
  "$TARGET_CFG/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: daily cadence is absent\n' >&2; exit 70; }
grep -q 'LANE1_STEP_ENTRY_DUMP' "$TARGET_CFG/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: entry writer is absent\n' >&2; exit 70; }
printf 'BUILD_DONE: %s\n' "$TARGET_CFG/BLD/bin/nemo.exe"
