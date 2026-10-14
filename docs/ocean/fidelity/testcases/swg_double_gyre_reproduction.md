# Reproducing the NEMO–legoESM shallow-water double gyre

This is a clean-machine runbook for a colleague who knows NEMO and Python but
not legoESM. It builds the `SWG_DG` case, proves the two-line case variant is
inert on shipped SWG, runs the 4,000 km eddying double gyre for three years in
both models, compares annual states, compares matched daily-mean climate
statistics, and makes a side-by-side movie. The workflow has nine numbered
steps; the import check is part of environment step 1.

The exact reference claim is deliberately narrow: on the pinned composition,
legoESM end-of-day `ssh_m`, `u_m_s`, and `v_m_s` at days 365, 730, and 1095
are IEEE-word identical to NEMO `sshn[0]`, `un[0,0]`, and `vn[0,0]` over the
T, U, and V wet masks. This is not full-restart identity and says nothing
about dry/storage cells.

## Before you start

You need:

- network access to the NEMO Forge, conda-forge, and Python wheel index;
- a clean legoESM checkout at this branch and a separate writable NEMO tree;
- GNU/Linux x86-64, conda, Git, patch, and SHA-256 tools;
- 32 CPU cores for NEMO and an idle CUDA GPU; the only tested GPU capacity is
  32 GiB, so a smaller card is unmeasured;
- at least 100 GiB free for outputs (the reference NEMO directory is
  56,985,134,712 bytes and the legoESM daily-mean directory is
  21,803,971,913 bytes); and
- a scheduling window of about 7 hours on 32 CPU cores and 2.5 hours on one
  GPU, plus storage-bound comparison/movie time.

Those reservations round upward from the measured 22,918 s NEMO and 8,117 s
GPU runs in `swg_double_gyre_reference_costs.json`. The current manifests and
logs, not those planning values, are authoritative for your machine. Never
pre-empt an occupied GPU.

## What you should see

The 5-minute case, 8/2-second rank arms, and 504-second smoke measurements are
bound to the fresh-run artifacts in the Round-C receipt. Long-run timings are
bound to `swg_double_gyre_reference_costs.json`.

| Step | Command below | Required final line | Reference time |
|---|---|---|---|
| Import check | Python environment check | `CHECKOUT_IMPORT_OK ...` | seconds |
| Build variant | `build_swg_dg.sh --execute` | `BUILD_DONE .../tests/SWG_DG` | not benchmarked |
| Case identity | `check_swg_dg_identity.sh --execute` | `SWG_DG_IDENTITY BIT_IDENTICAL variables=41 differing=0` | about 5 min in fresh operator run |
| Rank control | `swg_dg_rankcheck.sh --execute` | `all_bit_identical=True variables=21` | 1-rank 8 s; 32-rank 2 s |
| NEMO smoke | `swg_md_run.sh --smoke ... --execute` | `NEMO_RUN_COMPLETE label=dg_smoke steps=51840 ranks=32 ...` | about 8 min in fresh operator run |
| NEMO long run | `swg_md_run.sh --three-year ... --execute` | `NEMO_RUN_COMPLETE label=dg_3yr steps=1892160 ranks=32 ...` | 22,918 s (6.366 h) |
| legoESM long run | GPU wrapper | `COMPLETED days=1095 backend=gpu ...` | 8,117 s (2.255 h) |
| Annual checks | restart checker, three times | `SWG_DG_RESTART_IDENTITY PASS ... unequal=0 ...` | not benchmarked |
| Climate checks | climate instrument, twice | `PASS ...` | storage-bound; not benchmarked |
| Movie | movie tool | `SWG_DG_MOVIE COMPLETE frames=365 ...` | recorded in the command log |

“Not benchmarked” is intentional: this guide does not invent a cost where no
timing artifact exists.

## 1. Reconstruct the environments

Choose only paths on your machine:

```bash
export LEGOESM_ROOT=/path/to/clean/legoesm
export NEMO_ROOT=/path/to/new/nemo-5.0.2
export NEMO_PREFIX=/path/to/new/nemo-build-environment
export PYTHON_PREFIX=/path/to/new/python-3.13.0
export PYTHON_ENV=/path/to/new/legoesm-swg-dg-python
export REPRO=/path/to/new/swg-dg-reproduction
```

Clone and verify NEMO exactly:

```bash
git clone --branch 5.0.2 --single-branch \
  https://forge.nemo-ocean.eu/nemo/nemo.git "$NEMO_ROOT"
git -C "$NEMO_ROOT" checkout --detach \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
test "$(git -C "$NEMO_ROOT" rev-parse HEAD)" = \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
```

Recreate the exact linux-64 NEMO environment. Use the readable YAML only if
the explicit build file cannot be solved; that fallback is no longer an exact
build-composition reproduction.

```bash
conda create --prefix "$NEMO_PREFIX" --file \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/nemo-build-linux-64-explicit.txt"
export PATH="$NEMO_PREFIX/bin:$PATH"
```

The exact NEMO lock has 122 package/build URLs and an MD5 on every URL. Create
the exact Python 3.13.0 interpreter from conda-forge build
`h9ebbce0_101_cp313`, then make the isolated virtual environment:

```bash
conda create --prefix "$PYTHON_PREFIX" --override-channels \
  --channel conda-forge 'python=3.13.0=h9ebbce0_101_cp313' pip
"$PYTHON_PREFIX/bin/python" -c \
  'import sys; assert sys.version_info[:3] == (3, 13, 0)'
"$PYTHON_PREFIX/bin/python" -m venv "$PYTHON_ENV"
export PY="$PYTHON_ENV/bin/python"
"$PY" -m pip install -r \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/python-gpu-requirements.txt"
export JAX_ENABLE_X64=1
mkdir -p "$REPRO"
```

The 40 Python pins include JAX, jaxlib, CUDA plugin and PJRT 0.10.0, every
required CUDA wheel, `typing-extensions==4.15.0`, and
`certifi==2026.4.22`. Full values and the readable conda alternative are in
`swg_double_gyre_environment.md` and `.json`.

Every command below binds imports to the checkout containing the script. Its
first line names that checkout, interpreter, NumPy version, and source file.
Verify it before continuing:

```bash
cd "$LEGOESM_ROOT"
"$PY" scripts/run/run_swg_double_gyre.py --help
```

Required prefix:

```text
CHECKOUT_IMPORT_OK root=/your/legoesm ... numpy=2.4.4 ...
```

The tools refuse if `legoesm.ocean` resolves outside this checkout. They also
set CPU/GPU selection explicitly before importing JAX.

### Package self-test boundary

Run the same 18-file lane battery used for this package:

```bash
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "$PY" -m pytest -q \
  tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py \
  tests/ocean/unit/test_nemo_tsunami_card.py \
  tests/ocean/fidelity/test_nemo_swg_check_records.py \
  tests/ocean/fidelity/test_nemo_swg_dg_climate.py \
  tests/ocean/fidelity/test_nemo_swg_dg_composition.py \
  tests/ocean/fidelity/test_nemo_swg_dg_restart_identity.py \
  tests/ocean/fidelity/test_nemo_swg_dg_round6.py \
  tests/ocean/fidelity/test_nemo_swg_dg_round7.py \
  tests/ocean/fidelity/test_nemo_swg_geometry_gate.py \
  tests/ocean/fidelity/test_nemo_swg_ladder.py \
  tests/ocean/fidelity/test_nemo_tsunami_ladder.py \
  tests/ocean/fidelity/test_swg_dg_reproduction_guards.py \
  tests/ocean/fidelity/test_swg_dg_reproduction_shell.py \
  tests/ocean/fidelity/test_swg_double_gyre_movie.py \
  tests/ocean/unit/test_nemo_swg_card.py \
  tests/ocean/unit/test_pv_flux_ene.py \
  tests/ocean/unit/test_swg_swe_rk3_stage.py \
  tests/ocean/validation/test_step_gradient_matrix.py
```

This Round-D tree collects 370 tests. On a machine with only the repository,
expect **352 passed, 18 skipped**: four geometry tests need the external mesh;
twelve historical ladder tests need admitted stage/restart/month records; one
inventory test needs the preprocessed NEMO source tree; and one six-step GYRE
gradient row is unconditionally skipped for its registered LLVM section-memory
failure. Those 18 tests exercise external campaign evidence and are not guards
for the portable package. The fixture-backed build/run/guard/comparator tests
must run and may not silently skip. The reference-machine final count and log
are cited in the Round-D receipt.

## 2. Build `SWG_DG`

The builder starts from shipped `tests/SWG`, removes only `key_xios`, renders
the committed scalar-math architecture, and applies the committed two-line
`usrdef_zgr.F90` patch so the wet extent follows `rn_domsiz`.

```bash
TOOLS="$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre"
bash "$TOOLS/build_swg_dg.sh" \
  --nemo-root "$NEMO_ROOT" --conda-prefix "$NEMO_PREFIX" \
  --manifest-dir "$REPRO/build-manifest" --python "$PY"
bash "$TOOLS/build_swg_dg.sh" \
  --nemo-root "$NEMO_ROOT" --conda-prefix "$NEMO_PREFIX" \
  --manifest-dir "$REPRO/build-manifest" --python "$PY" --execute
(cd "$REPRO/build-manifest" && \
  sha256sum -c swg_dg_build_manifest.sha256)
```

Preflight ends `PREFLIGHT_OK`; the build ends `BUILD_DONE`. Preserve the
self-contained build manifest. It pins NEMO revision, compiler composition,
architecture, patch, every case source, and executable.

## 3. Prove the variant is inert at 2,000 km

```bash
bash "$TOOLS/check_swg_dg_identity.sh" \
  --nemo-root "$NEMO_ROOT" --arch-name swg-dg-scalarmath \
  --output "$REPRO/case-identity" --python "$PY"
bash "$TOOLS/check_swg_dg_identity.sh" \
  --nemo-root "$NEMO_ROOT" --arch-name swg-dg-scalarmath \
  --output "$REPRO/case-identity" --python "$PY" --execute
```

The required result is:

```text
SWG_DG_IDENTITY BIT_IDENTICAL variables=41 differing=0
```

It proves the variant reproduces all 41 shipped-case output variables over ten
steps when `rn_domsiz=2000000`.

## 4. Prove NEMO rank-count independence

```bash
bash "$TOOLS/swg_dg_rankcheck.sh" \
  --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output "$REPRO/rank-control" --python "$PY"
bash "$TOOLS/swg_dg_rankcheck.sh" \
  --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output "$REPRO/rank-control" --python "$PY" --execute
```

Expected final block:

```text
SWG-DG-ROUND6-NEMO-RANK-CHECK
all_bit_identical=True variables=21
```

The checker stitches rank tiles by coordinates and compares every word of all
21 restart variables. The 2026-10-13 fresh run initially failed after both
NEMO arms completed because its bare interpreter imported another checkout;
that measured packaging defect is fixed by the mandatory `--python` plus the
checkout-binding preamble.

## 5. Run NEMO smoke and three-year decks

The NEMO namelist patch and legoESM's `SWG_MADDISON_NAMELIST` are independent
definitions, not one source of truth. On completion, `swg_md_run.sh` feeds
the complete NEMO run directory to a cross-model guard. The guard resolves all
131 fields read by the compiled SWE engine. It takes a value from
`output.namelist.dyn` when NEMO echoes it; otherwise it applies NEMO's actual
`namelist_ref` then `namelist_cfg` override order to the two content-hashed run
inputs. It also compares the 16 case-specific physical values common to the
echo and the legoESM card, for 158 comparison rows in total. Only these two
output controls may differ:

- `sn_cfctl.l_runstat`: NEMO completion output with no legoESM analogue; and
- `nn_stock`: annual NEMO restart tiles versus daily legoESM NPZ output.

Any other physical, numerical, or run-control difference fails. NEMO's echo
omits the 21-field `nambdy` group even though the executable reads it. Those
values are therefore resolved from the hashed run inputs. `ocean.output` does
not print a machine-readable `nambdy` value on this case, and the report says
`not_printed` rather than claiming an unavailable cross-check.

```bash
NEMO_RUNS="$REPRO/nemo"
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --smoke dg_smoke 32
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --smoke dg_smoke 32 --execute

bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --three-year dg_3yr 32
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --three-year dg_3yr 32 --execute
export NEMO_RUN="$NEMO_RUNS/dg_3yr"
(cd "$NEMO_RUN" && sha256sum -c manifest.sha256)
```

The smoke is 51,840 steps. The three-year run is 1,892,160 steps with daily
means and restarts at 630,720, 1,261,440, and 1,892,160. Each command prints
`SWG_DG_DECK_GUARD PASS compiled_reads=131/131 comparison_rows=158
allowed_differences=2` before its final
`NEMO_RUN_COMPLETE` line. Retain `output.namelist.dyn`, `deck_guard.json`,
`run.log`, `run.stat`, `time.step`, and both manifests.

## 6. Run legoESM

Pick an idle physical GPU explicitly; the example uses device 0.

```bash
export LEGO_RUN="$REPRO/lego/dg_gpu_1095d_daily_mean"
PYTHON="$PY" "$LEGOESM_ROOT/scripts/run/run_swg_double_gyre_gpu.sh" \
  --gpu-device 0 --days 1095 --daily-means --output "$LEGO_RUN"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["device_memory_at_completion"]; assert p["status"]=="complete" and p["completed_days"]==1095 and p["wall_seconds"]>0 and m["supported"] and m["peak_bytes_in_use"]>0; print("GPU_MANIFEST PASS wall_seconds=%.3f peak_bytes_in_use=%d"%(p["wall_seconds"],m["peak_bytes_in_use"]))' \
  "$LEGO_RUN/manifest.json"
```

Required prefixes:

```text
COMPLETED days=1095 backend=gpu certified_policy=False wall_seconds=
GPU_MANIFEST PASS wall_seconds=... peak_bytes_in_use=...
```

The daily-mean run also writes end-of-day states. The reference wall time was
8,117 s; the fresh manifest is the source for your runtime and peak allocator
memory.

The debugging fallback is explicit CPU fp64/libm and uses the formal
materialized-step composition. It cannot create daily means and is not the GPU
composition used for annual identity:

```bash
"$PY" scripts/run/run_swg_double_gyre.py --mode cpu-certified \
  --days 1 --output "$REPRO/lego/dg_cpu_1d"
```

The measured one-day reference manifest records 233.93357684463263 s. Linear
extrapolation is about 71.2 hours for 1,095 days; this is an estimate, not a
measured long CPU run.

## 7. Compare annual completed states

The comparison is exactly:

| legoESM | NEMO | Instant and scope |
|---|---|---|
| `ssh_m` | `sshn[0]` | completed step N, T wet mask |
| `u_m_s` | `un[0,0]` | completed SWE layer velocity, U wet mask |
| `v_m_s` | `vn[0,0]` | completed SWE layer velocity, V wet mask |

NEMO `uu_n`/`vv_n` are zero barotropic work arrays here, not evidence of the
same capture instant. The checker hashes them for provenance but does not
compare them to layer velocities. It also refuses wrong steps/days, incomplete
manifests, wrong masks/decks/dtypes, and non-finite data.

```bash
CHECK="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_restart_identity.py"
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

The pinned reference composition prints:

```text
SWG_DG_RESTART_IDENTITY PASS day=365 step=630720 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
SWG_DG_RESTART_IDENTITY PASS day=730 step=1261440 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
SWG_DG_RESTART_IDENTITY PASS day=1095 step=1892160 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
```

The three committed reports beside this guide are the citable artifacts for
that statement. They prove 1,227,520 wet words at each checkpoint, not a full
restart or dry-cell comparison.

## 8. Compare matched daily-mean climate

```bash
CLIMATE="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_climate.py"
JAX_PLATFORMS=cpu "$PY" "$CLIMATE" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 1 --days 365 --output "$REPRO/climate_year1"
JAX_PLATFORMS=cpu "$PY" "$CLIMATE" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 366 --days 730 --output "$REPRO/climate_years2_3"
```

Expected lines are `PASS year 1, spin-up -> ...` and
`PASS years 2-3 -> ...`. On the pinned reference trajectory, maximum absolute
differences were:

| Window | Basin-mean KE | Basin-mean EKE |
|---|---:|---:|
| days 1–365 | `2.286230926795696e-11` | `1.1921534939784983e-11` |
| days 366–1095 | `2.2225409013199027e-11` | `1.402270660699756e-11` |

Across both windows, the measured maxima are
`2.286230926795696e-11` for kinetic energy and
`1.402270660699756e-11` for eddy kinetic energy.
These are measured reference values, not a cross-toolchain acceptance bar.
NEMO daily means are fp32, and no sampling-uncertainty/post-decorrelation bar
has been registered. Retain and report every metric.

## 9. Render the matched movie

```bash
export MPLCONFIGDIR="$REPRO/matplotlib-cache"
JAX_PLATFORMS=cpu "$PY" \
  "$LEGOESM_ROOT/scripts/plot/swg_double_gyre_movie.py" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 1 --days 365 --fps 12 \
  --output "$REPRO/swg_dg_year1.mp4"
```

Expected line:

```text
SWG_DG_MOVIE COMPLETE frames=365 output=.../swg_dg_year1.mp4
```

The two rows are NEMO and legoESM; columns are relative vorticity/f, SSH, and
speed with shared per-frame scales. The tool verifies/hashes its inputs,
starts ffmpeg before JAX threads, streams frames, and atomically publishes the
MP4. It never presents a partial encode as complete.

## If annual identity fails: make and score steps 1–10

Do not change a tolerance. Make a one-rank NEMO restart at every step:

```bash
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$REPRO/nemo" --python "$PY" --short-ladder dg_ladder
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$REPRO/nemo" --python "$PY" --short-ladder dg_ladder --execute
```

Score the same outer-scan composition used by the long GPU driver, first on
CPU for a portable diagnostic:

```bash
LADDER="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_short_ladder.py"
"$PY" "$LADDER" --nemo-root "$REPRO/nemo/dg_ladder" \
  --backend cpu --composition outer-scan \
  --json "$REPRO/short_ladder_outer_scan.json"
```

On the reference machine this prints 30 zero-unequal rows and ends:

```text
SWG_DG_SHORT_LADDER EXACT composition=outer-scan backend=cpu first_difference=NONE
```

Score the operational device composition with a separate output path:

```bash
"$PY" "$LADDER" --nemo-root "$REPRO/nemo/dg_ladder" \
  --backend gpu --gpu-device 0 --composition outer-scan \
  --json "$REPRO/short_ladder_outer_scan_gpu.json"
```

The report identifies the first differing step/field, unequal words, maximum
residual, region, hashes, and a binding one-ULP control. Never reuse the CPU
JSON path; the scorer refuses overwrites.

The formal materialized composition is a useful discriminator and is expected
to return exit 1 on the reference composition:

```bash
"$PY" "$LADDER" --nemo-root "$REPRO/nemo/dg_ladder" \
  --backend cpu --composition materialized \
  --json "$REPRO/short_ladder_materialized.json" || test "$?" -eq 1
```

Its first completed-state difference is:

```text
SWG_DG_SHORT_LADDER DEBT composition=materialized backend=cpu first_difference=step2.ssh_m bits=189
```

Compare both reports with the committed reference behavior:

```bash
"$PY" "$LADDER" --compare-reference \
  --outer-report "$REPRO/short_ladder_outer_scan.json" \
  --materialized-report "$REPRO/short_ladder_materialized.json"
```

The required reference-machine lines are:

```text
SWG_DG_SHORT_LADDER_BASELINE outer-scan same as reference
SWG_DG_SHORT_LADDER_BASELINE materialized same as reference
```

On another composition, the command instead prints the first step and field
whose counts or maximum residual depart from the reference. To compare the GPU
form, substitute `short_ladder_outer_scan_gpu.json` for `--outer-report`.

Three often-confused debt counts describe different reductions:

- **189** is the number of unequal wet SSH words in the materialized
  completed state at step 2. The short-ladder command above produces it.
- **1,278** is the number of unequal wet SSH words between legoESM's
  materialized stage-1 output and NEMO's separately instrumented stage-2 entry
  at step 2. Given an admitted instrumented stage root, this command produces
  it:

  ```bash
  ROUND7="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round7.py"
  export STAGE_ROOT=/path/to/instrumented/nemo/step2/stage-entry-records
  "$PY" "$ROUND7" --root "$STAGE_ROOT" \
    --json "$REPRO/formal_stage_entry_discriminator.json"
  ```

- **414,545** is the step-10 sum of materialized completed-state unequal words
  over SSH, U, and V (`278,318 + 3,143 + 133,084`). It is the accumulated
  endpoint of the same materialized-program debt, not a third comparison
  boundary. This command produces that total:

  ```bash
  COMPOSITION="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_composition.py"
  "$PY" "$COMPOSITION" --root "$REPRO/nemo/dg_ladder" \
    --json "$REPRO/step10_composition_discriminator.json"
  ```

The committed compact baseline records all 30 rows for both program forms.
The 640 V-stress forcing words differing only by signed zero also remain
reported, not excepted, pending the user's exactness-policy decision.

## Honest expectation across machines

Bit identity depends on the physical transcription together with GNU Fortran
version and flags, the scalar-math architecture, `key_nosignedzero`, NEMO rank
composition (controlled above), JAX/jaxlib/XLA versions, and the GPU versus CPU
program composition. The compiled-day outer scan matches NEMO on this deck;
the formal per-step program differs in the last bit. On shipped SWG, the
opposite composition choice was the certified exact one.

A compiler, XLA, or GPU change may therefore introduce a last-bit difference.
This flow is chaotic, so individual eddies can decorrelate within weeks. After
that, annual word comparison is not informative; the matched daily-mean
climate statistics are the comparison. They should remain close, but no
cross-toolchain acceptance bar exists yet.

If identity fails, report the exact environment/build/run manifests, first
failing annual JSON, 1–10-step outer-scan JSON (and GPU device), materialized
discriminator JSON, and both climate reports. That evidence separates deck,
capture alignment, compiler, and composition differences without moving a
bar.
