#!/bin/bash
# Build the gSAM 1.8.8 P3 oracle driver used by make_fixture.py.
#
# Usage: bash build.sh <gsam1.8.8.tar.gz> <build_dir>
#
# The tarball is http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz
# (sha256 0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c; gSAM
# has no git tag/SHA, the tarball hash is the pin).  Extracts the P3 sources,
# the lookup table and the three case soundings into <build_dir>/src, applies
# patch_p3.py (legoESM saturation + diag dumps + compile fixes) and compiles
# p3_driver + qsat_check with gfortran in fp64 (-fdefault-real-8
# -fdefault-double-8), exactly as the tier-2 p3_microphysics audit did.
# Compute: run inside a SLURM job, not on a login node.
set -euo pipefail
TAR=$(readlink -f "$1"); B=$(readlink -m "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
echo "0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c  $TAR" | sha256sum -c -
mkdir -p "$B/src"
tar xzf "$TAR" -C "$B/src" --strip-components=1 \
  gSAM1.8.8/SRC/MICRO_P3/module_mp_p3.f90 gSAM1.8.8/SRC/MICRO_P3/micro_params.f90 \
  gSAM1.8.8/SRC/sat.f90 gSAM1.8.8/RUNDATA/p3_lookup_table_1.dat-v4.1 \
  gSAM1.8.8/CASES/RCE/snd_300K gSAM1.8.8/CASES/DYCOMS_RF01/snd gSAM1.8.8/CASES/MC3E/snd
python3 "$HERE/patch_p3.py" "$B/src/SRC/MICRO_P3" "$B"
gfortran --version | head -1
FF="-O0 -g -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -fno-range-check -fbacktrace -cpp -I$HERE"
cd "$B"
gfortran $FF -c micro_params_lego.f90
gfortran $FF -c "$HERE/stubs_lego.F90"
gfortran $FF -c src/SRC/sat.f90
gfortran $FF -c module_mp_p3_lego.f90
gfortran $FF -c "$HERE/p3_driver.F90"
gfortran $FF -o p3_driver p3_driver.o module_mp_p3_lego.o sat.o stubs_lego.o micro_params_lego.o
gfortran $FF -c "$HERE/qsat_check.F90"
gfortran $FF -o qsat_check qsat_check.o stubs_lego.o
test -x p3_driver && test -x qsat_check && echo BUILD_OK
