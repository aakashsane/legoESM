#!/bin/bash
# Build the CAM6 MG2 oracle drivers against UNMODIFIED CAM sources.
# usage: build.sh <CAM checkout (cam_cesm2_1_rel_60)> <build dir>
# The legoESM-qsat stub (wv_sat_legoesm.F90) replaces CAM's wv_sat_methods so
# both sides use the same saturation curve; micro_mg_utils.F90 and
# micro_mg2_0.F90 compile as shipped.  gfortran 8.5.0 was used for the
# committed fixture (tests/atmosphere/microphysics/oracle_fixtures/).
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$HERE/wv_sat_legoesm.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_driver "$HERE/mg2_driver.F90" wv_sat_legoesm.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_procs "$HERE/mg2_procs_driver.F90" wv_sat_legoesm.o micro_mg_utils.o micro_mg2_0.o
echo BUILD_OK
