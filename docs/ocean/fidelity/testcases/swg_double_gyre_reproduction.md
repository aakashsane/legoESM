# Reproducing the NEMO–legoESM shallow-water double gyre

This runbook rebuilds the NEMO `SWG_DG` case, proves that the case variant is
inert on the shipped 2,000 km deck, runs the 4,000 km double gyre in NEMO and
legoESM, compares the completed states at years 1–3, computes matched
daily-mean climate diagnostics, and renders a side-by-side movie. It assumes
familiarity with NEMO and Python, but not with legoESM.

The reference environment is pinned in
`swg_double_gyre_environment.md` and
`swg_double_gyre_environment.json`. Bit identity is an environment-sensitive
result; read “What exact agreement means” before interpreting a mismatch.

## Prerequisites and paths

Use a clean checkout of this branch and a clean NEMO 5.0.2 checkout at source
revision `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`. The NEMO environment must
provide GNU Fortran, Open MPI, NetCDF-C/Fortran and HDF5. XIOS is neither built
nor used. The commands below deliberately have no developer-machine defaults.

```bash
export LEGOESM_ROOT=/path/to/legoesm
export NEMO_ROOT=/path/to/nemo_5.0.2
export NEMO_PREFIX=/path/to/activated/nemo-conda-environment
export REPRO=/path/to/new/swg-dg-reproduction
export PY=/path/to/python
export PATH="$NEMO_PREFIX/bin:$PATH"

cd "$LEGOESM_ROOT"
"$PY" -m pip install -e '.[dev]'
"$PY" -m pip install \
  'numpy==2.4.4' 'equinox==0.13.7' \
  'jax==0.10.0' 'jaxlib==0.10.0' 'jax-cuda12-plugin==0.10.0' \
  'imageio-ffmpeg==0.6.0'
export PYTHONPATH="$LEGOESM_ROOT/packages/core:$LEGOESM_ROOT/packages/ocean:$LEGOESM_ROOT/packages/atmosphere:$LEGOESM_ROOT/packages/coupler:$LEGOESM_ROOT/packages/ice:$LEGOESM_ROOT/packages/land:$LEGOESM_ROOT/packages/ml:$LEGOESM_ROOT/packages/tools:$LEGOESM_ROOT/src"
export JAX_ENABLE_X64=1
mkdir -p "$REPRO"
```

Before the long run, compare your versions with the pin:

```bash
git -C "$NEMO_ROOT" rev-parse HEAD
"$NEMO_PREFIX/bin/gfortran" --version | head -1
"$NEMO_PREFIX/bin/mpirun" --version | head -1
"$PY" -c 'import importlib.metadata as m; import equinox,imageio_ffmpeg,jax,jaxlib,numpy; versions=(jax.__version__,jaxlib.__version__,m.version("jax-cuda12-plugin"),numpy.__version__,equinox.__version__,imageio_ffmpeg.__version__); print(*versions); assert versions == ("0.10.0","0.10.0","0.10.0","2.4.4","0.13.7","0.6.0")'
uname -r
```

## 1. Build the `SWG_DG` NEMO variant

The build tool starts from shipped `tests/SWG`, removes `key_xios`, installs
the committed reference architecture, and applies only the committed two-line
`usrdef_zgr.F90` patch: the wet-basin upper limits follow `rn_domsiz`. It
refuses an existing target, architecture, or manifest directory.

```bash
TOOLS="$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre"
bash "$TOOLS/build_swg_dg.sh" \
  --nemo-root "$NEMO_ROOT" --conda-prefix "$NEMO_PREFIX" \
  --manifest-dir "$REPRO/build-manifest"
bash "$TOOLS/build_swg_dg.sh" \
  --nemo-root "$NEMO_ROOT" --conda-prefix "$NEMO_PREFIX" \
  --manifest-dir "$REPRO/build-manifest" --execute
(cd "$REPRO/build-manifest" && \
  sha256sum -c swg_dg_build_manifest.sha256)
```

The preflight ends `PREFLIGHT_OK`. The build ends
`BUILD_DONE .../tests/SWG_DG`. Preserve the build-manifest directory; its
`artifacts/` copy makes the SHA-256 inventory self-contained and directly
checkable with the command recorded in `swg_dg_build_manifest.txt`. It pins
the binary, every case source, patch, architecture, source revision and
compiler composition.

## 2. Prove that the variant is inert on shipped SWG

At `rn_domsiz=2000000`, the patched domain limit is the same limit used by the
shipped case. Build a clean shipped reference and run both binaries for ten
steps on one rank:

```bash
bash "$TOOLS/check_swg_dg_identity.sh" \
  --nemo-root "$NEMO_ROOT" --arch-name swg-dg-scalarmath \
  --output "$REPRO/case-identity" --python "$PY"
bash "$TOOLS/check_swg_dg_identity.sh" \
  --nemo-root "$NEMO_ROOT" --arch-name swg-dg-scalarmath \
  --output "$REPRO/case-identity" --python "$PY" --execute
```

The required terminal line is exactly:

```text
SWG_DG_IDENTITY BIT_IDENTICAL variables=41 differing=0
```

This command calls `makenemo` and MPI only with `--execute`.

## 3. Check rank-count identity

The control runs the ten-step double-gyre deck on 1 and 32 ranks, stitches all
32 restart tiles by coordinates, and compares all 21 restart variables as
IEEE words. It is not a three-field spot check.

```bash
bash "$TOOLS/swg_dg_rankcheck.sh" \
  --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" --output "$REPRO/rank-control" \
  --python "$PY"
bash "$TOOLS/swg_dg_rankcheck.sh" \
  --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" --output "$REPRO/rank-control" \
  --python "$PY" --execute
```

Expected final lines:

```text
SWG-DG-ROUND6-NEMO-RANK-CHECK
all_bit_identical=True variables=21
```

## 4. Run NEMO smoke and three-year integrations

Both modes apply the same committed namelist patch. It is the single source of
truth for the 4,000 km basin, 6.25 km spacing, 50 s step, 0.10 N m-2 wind,
9.80665 m s-2 gravity, 10 m2 s-1 viscosity and 1e-7 s-1 drag. The 30-day smoke
uses 51,840 steps. The three-year run uses 1,892,160 steps, daily means and
annual restarts at steps 630,720, 1,261,440 and 1,892,160.

```bash
NEMO_RUNS="$REPRO/nemo"
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --smoke dg_smoke 32
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --smoke dg_smoke 32 --execute

bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --three-year dg_3yr 32
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --three-year dg_3yr 32 --execute
NEMO_RUN="$NEMO_RUNS/dg_3yr"
(cd "$NEMO_RUN" && sha256sum -c manifest.sha256)
```

The smoke and long runs end respectively with lines beginning
`NEMO_RUN_COMPLETE label=dg_smoke steps=51840 ranks=32` and
`NEMO_RUN_COMPLETE label=dg_3yr steps=1892160 ranks=32`. Inspect `time.step`,
`run.stat`, `run.log`, `manifest.sha256`, and `resolved_namelist_lines.txt`
before continuing.

## 5. Run legoESM

The GPU number is an explicit physical device selection and is recorded with
the visible JAX devices in `manifest.json`. Replace `0` with either idle GPU;
never pre-empt another job. A single daily-mean run also writes each day's
instantaneous endpoint, so it serves both the annual identity and climate
comparisons.

```bash
PYTHON="$PY" "$LEGOESM_ROOT/scripts/run/run_swg_double_gyre_gpu.sh" \
  --gpu-device 0 --days 1095 --daily-means \
  --output "$REPRO/lego/dg_gpu_1095d_daily_mean"
```

The required prefix is:

```text
COMPLETED days=1095 backend=gpu certified_policy=False wall_seconds=
```

On the reference V100S this took about 45 minutes per model year and 2.2 hours
for three years. NEMO took about two hours per model year on 32 Xeon 6242R
cores and 6.4 hours for three years. Both commands record their own wall time;
use those values rather than assuming reference throughput.

The CPU fallback runs the formal fp64/libm materialized-step composition and
does not support daily means. It is useful for smoke/debug work, not for
reproducing the GPU composition's exact annual endpoint:

```bash
JAX_PLATFORMS=cpu "$PY" \
  "$LEGOESM_ROOT/scripts/run/run_swg_double_gyre.py" \
  --mode cpu-certified --days 1 --output "$REPRO/lego/dg_cpu_1d"
```

The reference run printed:

```text
COMPLETED days=1 backend=cpu certified_policy=True wall_seconds=233.934 deck_certification=reference-evidence-composition-dependent output=.../dg_cpu_1d
```

On the reference Xeon host, the measured full-resolution one-day fallback
took `233.934` seconds. A straight-line 1,095-day extrapolation is about 71.2
hours, so it is not a practical substitute for the GPU run. Read the exact
measurement from `manifest.json` fields `wall_seconds` and
`wall_seconds_per_model_day`.

## 6. Compare years 1, 2 and 3

The checker compares every wet word of these completed-step fields:

| legoESM snapshot | NEMO restart | Instant |
|---|---|---|
| `ssh_m` | `sshn[0]` | after completed step N |
| `u_m_s` | `un[0,0]` | completed SWE layer velocity after step N |
| `v_m_s` | `vn[0,0]` | completed SWE layer velocity after step N |

NEMO `uu_n`/`vv_n` are zero barotropic workspace arrays in these restarts;
the checker hashes and reports them but does not misrepresent them as
same-instant layer velocities. It verifies each restart `kt`, the legoESM
day/step, all content manifests, masks, unequal-word counts, maximum absolute
difference and signed ULP histograms.

```bash
CHECK="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_restart_identity.py"
LEGO_RUN="$REPRO/lego/dg_gpu_1095d_daily_mean"
NEMO_RUN="$REPRO/nemo/dg_3yr"

JAX_PLATFORMS=cpu "$PY" "$CHECK" --nemo-root "$NEMO_RUN" \
  --legoesm-npz "$LEGO_RUN/day0365.npz" --day 365 --step 630720 \
  --json "$REPRO/year1_restart_identity.json"
JAX_PLATFORMS=cpu "$PY" "$CHECK" --nemo-root "$NEMO_RUN" \
  --legoesm-npz "$LEGO_RUN/day0730.npz" --day 730 --step 1261440 \
  --json "$REPRO/year2_restart_identity.json"
JAX_PLATFORMS=cpu "$PY" "$CHECK" --nemo-root "$NEMO_RUN" \
  --legoesm-npz "$LEGO_RUN/day1095.npz" --day 1095 --step 1892160 \
  --json "$REPRO/year3_restart_identity.json"
```

On the reference environment the exact pass lines are:

```text
SWG_DG_RESTART_IDENTITY PASS day=365 step=630720 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
SWG_DG_RESTART_IDENTITY PASS day=730 step=1261440 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
SWG_DG_RESTART_IDENTITY PASS day=1095 step=1892160 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
```

The three citable reference reports are committed as
`swg_dg_year{1,2,3}_restart_identity.json` beside this runbook.

## 7. Run the daily-mean climate instrument

The instrument refuses snapshot-versus-mean sampling, incomplete daily
sequences, unpinned content, wrong clocks, or a different physical deck. Run
the spin-up year and the years 2–3 window separately:

```bash
CLIMATE="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_climate.py"
JAX_PLATFORMS=cpu "$PY" "$CLIMATE" --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 1 --days 365 --output "$REPRO/climate_year1"
JAX_PLATFORMS=cpu "$PY" "$CLIMATE" --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 366 --days 730 --output "$REPRO/climate_years2_3"
```

Expected pass lines are:

```text
PASS year 1, spin-up -> .../climate_year1
PASS years 2-3 -> .../climate_years2_3
```

The reference reports put basin-mean KE/EKE maximum absolute differences at
order 1e-11. Because the NEMO daily fields are stored in fp32, spatial-map
metrics are not expected to be bit-identical; inspect every metric in
`swg_dg_climate_report.json` rather than treating the successful `PASS` line
as a numerical tolerance gate.

## 8. Render the side-by-side movie

The movie uses matched daily means and shared per-frame colour scales. It has
NEMO and legoESM rows and relative vorticity/f, SSH and speed columns.

```bash
JAX_PLATFORMS=cpu "$PY" "$LEGOESM_ROOT/scripts/plot/swg_double_gyre_movie.py" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 1 --days 365 --fps 12 \
  --output "$REPRO/swg_dg_year1.mp4"
```

Expected line:

```text
SWG_DG_MOVIE COMPLETE frames=365 output=.../swg_dg_year1.mp4
```

`imageio-ffmpeg` supplies ffmpeg; the script streams frames and refuses to
overwrite an existing movie.

## What exact agreement means

Bit-for-bit agreement was obtained with the pinned toolchain above. It is a
property of the physical transcription *and* the compiled program
composition, not a promise across compilers, JAX/XLA releases, CPUs, or GPUs.
The committed ten-step composition discriminator makes this concrete:

```bash
JAX_PLATFORMS=cpu "$PY" \
  "$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_composition.py" \
  --root "$REPRO/rank-control/np1" \
  --json "$REPRO/composition_discriminator.json"
```

The reference output is:

```text
SWG-DG-TEN-STEP-COMPOSITION-DISCRIMINATOR
outer_jit_lax_scan: prognostic_bit_unequal=0 all_bit_unequal=640 signed_zero_only=640
materialized_python_loop: prognostic_bit_unequal=414545 all_bit_unequal=415185 signed_zero_only=640
control_binding=True
```

On the reference toolchain, the one-outer-JIT ten-step scan has zero unequal
prognostic words against NEMO, while retaining the reported 640 V-stress
signed-zero words. The formal materialized loop has 414,545 unequal
prognostic words at step 10. Its first dynamic difference appears at step 2;
the stage ladder reports 1,278 wall-adjacent SSH words in the last bit. No bar
is relaxed. The forcing sign bits remain DEBT pending the user's policy
ruling. The result and binding one-ULP plant are committed in
`swg_dg_composition_discriminator.json`.

The shipped 2,000 km deck shows the complementary warning: its certified
materialized ladder is exact to NEMO, while an outer scan differs from that
production composition in the last bits. See the standalone replay receipt,
the double-gyre round-7 receipt, and the SWG gradient-gate round-2 receipt.
Program composition must therefore be recorded, never inferred from a shared
Python function alone.

On a different compiler, JAX version, or GPU, annual word identity may fail.
That is not surprising in a chaotic eddying flow: last-bit changes cause
individual eddies to decorrelate within weeks. After decorrelation, the
meaningful comparison is the matched-sampling climate instrument. Compare and
report every statistic, but do not apply the reference run's order-1e-11
difference as a cross-toolchain acceptance bar: no sampling-uncertainty or
post-decorrelation bar has yet been registered.

If exact identity fails, retain all manifests and report:

1. the complete environment JSON and generated build/run manifests;
2. the first failing annual report and its field/ULP histogram;
3. a short 1–10-step NEMO restart ladder and the first differing step;
4. the ten-step composition-discriminator report; and
5. both climate reports.

Do not change an exactness bar to make a new toolchain pass. The useful
question is whether the first difference comes from the deck, a missing
field/capture alignment, or compiler composition.
