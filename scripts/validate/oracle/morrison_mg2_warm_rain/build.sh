#!/bin/bash
# Build the CAM6 MG2 oracle drivers against UNMODIFIED CAM sources.
# usage: build.sh <CAM checkout (cam_cesm2_1_rel_60)> <build dir>
# CAM's own wv_sat_methods.F90 (Goff-Gratch, initialised by the drivers with
# legoESM's epsilon), micro_mg_utils.F90 and micro_mg2_0.F90 compile as
# shipped.  gfortran 8.5.0 was used for the
# committed fixture (tests/atmosphere/microphysics/oracle_fixtures/).
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_driver "$HERE/mg2_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_procs "$HERE/mg2_procs_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
# Probe build (S10.2 stage oracle): a COPY of micro_mg2_0.F90 that only records
# nctend right after the activation re-apply (micro_mg2_0.F90:2022) in a public
# module array; no MG2 arithmetic changes.  Built in tap/ so the unmodified
# objects above stay untouched.
mkdir -p tap
sed -e '0,/^contains$/s//real(r8), public :: nctend_stage(1,512) = 0._r8\ncontains/' \
    -e 's/^  nctend = nctend + npccn$/&\n  nctend_stage = 0._r8\n  nctend_stage(1:mgncol,1:nlev) = nctend/' \
    "$CAM/micro_mg2_0.F90" > tap/micro_mg2_0.F90
test "$(grep -c nctend_stage tap/micro_mg2_0.F90)" -eq 3
test "$(diff "$CAM/micro_mg2_0.F90" tap/micro_mg2_0.F90 | grep -c '^>')" -eq 3
(cd tap && gfortran $FF -I.. -c "$CAM/micro_mg_utils.F90" micro_mg2_0.F90 \
   && gfortran $FF -DMG2_TAP -I. -o ../mg2_tap "$HERE/mg2_driver.F90" ../wv_sat_methods.o \
      micro_mg_utils.o micro_mg2_0.o)
echo BUILD_OK
