# NEMO testcase L2 GYRE high resolution — round 11 receipt

Date: 2026-10-16

Incoming lane tip: `854d2da150da603813f2637e8ae764a7c5459ccf`.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round11/`.
The exact-rung oracle is the admitted scalar-math step-2 record from round 7.
The climate reference is certified vector-math NEMO; scalar-math NEMO is the
labelled, non-certified twin floor.  The candidate is the registered
`post_round9_219db484b` five-year legoESM run.

## Verdict

**The remaining entry-`kt=2` tracer boundary is a measured coupled
rounding-order/fusion debt, not a transcription owner.  No shared-path change
is warranted.**  The exhaustive full-step bisection found two equivalent
three-member temperature minima: formed diffusivity + Kaa thickness + formed
temperature content, or pre-formation heat diffusivity + Kaa thickness +
formed temperature content.  Salinity has the analogous minima, and joint
T/S exactness additionally needs formed salinity content.  Every proper
one-member removal fails the zero-word bar.

The day-30..180 excess over the scalar/vector NEMO twin is spatially
structured and algebraic under the preregistered fit, not exponential.  All
six stored fields already exceed four times the twin floor at day 30, so the
daily, float32 NEMO output cannot distinguish a uniquely first field.  SST is
the registered tie-breaker because the NEMO twin SST difference is still
zero at day 30; its largest signed excess is at the eastern boundary.  Bottom
velocity is unmeasured because the registered legoESM daily files store only
surface and external-mode velocities.

## Finding 1 — exact minimal subset

**Finding.**  Production at the scalar step-1 vertical-solve boundary differs
from NEMO in 110,286 temperature words (maximum
`1.0658141036401503e-14 K`, 3 ULP) and 103,327 salinity words (maximum
`2.1316282072803006e-14`, 3 ULP).  The instrument first substituted one
recorded input at a time, then exhausted all 31 non-empty subsets of the five
arrays consumed by the production tracer dispatcher and all 63 non-empty
subsets of those arrays plus upstream heat diffusivity in the unobserved,
compiled full step.  Its disabled seam is bit-exact and all six active plants
fire.

The isolated production tracer dispatcher has the source-ordered minima one
would expect:

| scored result | smallest exact consumed-input set |
|---|---|
| temperature | formed K + temperature content |
| salinity | formed K + salinity content |
| joint T/S | formed K + temperature content + salinity content |

The full fused step requires one additional selected leaf:

| scored result | smallest exact full-step sets |
|---|---|
| temperature | formed K + Kaa thickness + T content; **or** heat K + Kaa thickness + T content |
| salinity | formed K + Kaa thickness + S content; **or** heat K + Kaa thickness + S content |
| joint T/S | formed K + Kaa thickness + T content + S content; **or** heat K + Kaa thickness + T content + S content |

For the first joint minimum, removing each member gives:

| member removed | T maximum / unequal | S maximum / unequal |
|---|---:|---:|
| formed K | `1.0658e-14` / 1,774 | `1.4211e-14` / 1,443 |
| Kaa thickness | `1.0658e-14` / 64,779 | `1.4211e-14` / 61,309 |
| T content | `7.1054e-15` / 67,325 | `0` / 0 |
| S content | `0` / 0 | `1.4211e-14` / 63,449 |

Kaa thickness is bit-identical to NEMO on all 288,000 active words, and the
isolated dispatcher does not need it.  Merely selecting that identical leaf
is nevertheless necessary in the fused full-step minimum.  Formed K and heat
K are interchangeable here because the recorded step-1 isoneutral addition
is zero; neither is a single owner.  These facts discriminate a real
floating-point composition/fusion coupling from a wrong input value or a
transcribed physical operator.

**Disposition.**  Classified `COUPLED_ROUNDING_ORDER_DEBT`; stopped without a
physics change.  Forcing full-step bit identity by rewriting a shared kernel
would tune compiler composition to this probe rather than correct NEMO
arithmetic.

**Evidence.**  `step1_vertical_minimal_subset_v4.json`, SHA-256
`a3a3fdefc81c1fd31f6b9af64e246b7ae9fcfc64f47684c7e5eeacc3088d32e8`,
produced from clean instrument commit `f108006d1`.  The admitted oracle
manifest is `28e73955382c80179e2e244294ccc28dd4a67762fe826d9da19f887580b6d1f0`.

## Finding 2 — day-30..180 excess over the twin envelope

**Finding.**  The extended divergence instrument first reproduces the
independent NEMO-twin SSH checkpoints exactly at their printed precision:
`1.04e-9`, `1.17e-9`, `1.19e-8`, `1.21e-6`, `2.95e-4`, and `3.89e-3` on days
30, 60, 90, 120, 180, and 365.  Its exact and one-ULP controls separate.

At day 30 every measured field already clears the preregistered four-times
twin criterion:

| field | legoESM normalized RMS | twin normalized RMS | ratio |
|---|---:|---:|---:|
| depth-mean U | `2.3434e-7` | `4.2995e-10` | `545` |
| depth-mean V | `1.9657e-7` | `5.6301e-10` | `349` |
| SSH | `1.5270e-7` | `1.0419e-9` | `147` |
| SST | `6.8570e-7` | `0` | unresolved/infinite |
| surface U | `1.6144e-6` | `1.6442e-9` | `982` |
| surface V | `1.2193e-5` | `2.3296e-9` | `5,234` |

Thus “which field diverges first” is a six-way tie at the first registered
day, not an identifiable ordering.  NEMO's daily fields are float32, so the
zero SST twin floor is a storage-resolution bound rather than proof that the
two NEMO trajectories are physically identical.  By day 180 the legoESM/twin
ratios are `7.16`, `5.64`, `8.20`, `6.10`, `3.35`, and `2.53` in the table's
field order.

The SST tie-breaker's signed candidate-minus-twin squared-error excess over
days 30..180 is `19.17`, `31.29`, `12.97`, `24.38`, and `78.17` from western
boundary through eastern boundary.  The largest contribution is therefore
the eastern boundary, not a localized western-boundary seed.  The basin SST
fit favors algebraic growth (log-space RMS residual `0.852`) over exponential
growth (`1.190`); the eastern-boundary fit agrees (`1.971` versus `2.538`).
Every other basin field also favors algebraic growth.  Under the registered
interpretation, the day-30..180 excess behaves as a persistent structured
difference; the measurement does not support calling that interval
chaos-dominated.

Depth-mean U/V are reported separately using NEMO's static z-coordinate
thickness-weighted daily 3-D velocities and legoESM's stored external-mode
daily means.  Bottom velocity remains `UNMEASURED`: no matching legoESM
bottom field exists in the registered run.

**Disposition.**  Recorded as diagnostic debt, not a climate failure.  The
same run still passes the preregistered year-5 climate bars against both NEMO
builds, and this round supplies no single shared-path owner.

**Evidence.**
`divergence_regional_v2/gyre_hires_divergence_report.json`, SHA-256
`7e08737b206723df56fd9ca5dcf46053e143b188f16be7bdcb2eb101101beb54`,
and `gyre_hires_divergence_arrays.npz`, SHA-256
`d96f9cf31ec96d598d05418ce1835dbedc64b2e49c1f4be171c44913e1eb48b6`,
produced by instrument commit `d42b00bc1`.

## Shared-path disposition and next measurement

No shared-path change is warranted.  The only model change is a private,
non-configurable diagnostic seam exposing salinity content alongside
temperature content; disabled-seam identity is exact.  No deck value,
physical operator, forcing, or production configuration changed.

The single next measurement is the entry-`kt=3` SSH boundary already open in
round 10: score the existing scalar step-2 RK stage records to identify the
first stage that moves SSH, then acquire step-2 barotropic substeps only if
the first differing stage lies inside `dyn_spg_ts`.  No acquisition is needed
before that stage-boundary score.

## Inherited pins / OPEN

Inherited unchanged: the shipped one-degree digest `99b2dfce324b0f77`, the
one-degree no-worsening ceiling, `r4` and `r4_smag` decks, 31 levels, RK3,
split-explicit free surface, TKE/EVD, Redi tracers, 360-day forcing clock,
scalar-libm CPU certification, zero-word exact-rung bar, float32 NEMO daily
storage, thickness-weighted NEMO daily temperature, registered long-run
manifests, and explicit physical-GPU selection.

OPEN, in order:

1. Localize the existing entry-`kt=3` SSH debt by scalar step-2 RK stage.
2. If it first appears inside the external mode, acquire scalar step-2
   barotropic substeps; the current scalar step-2 record has substeps for
   step 1 only.
3. Bottom-regional divergence remains unmeasured for this completed run.
4. GitHub issue #1455 could not be updated from this sandbox because network
   access to `api.github.com` was unavailable.

## Tests and controls

The final verification battery and citation-gate results are recorded in the
landing commit below.  Controls exercised by the committed instruments are:
disabled full-step seam identity, six active input plants, exhaustive 31- and
63-subset inventories, NEMO-twin benchmark replay, exact divergence identity,
one-ULP divergence detection, five wet-span region placement, and synthetic
exponential/algebraic growth separation.

## Compiled-source citations

The executed scalar writer zeros the surface coefficient and forms the
source-ordered lower, upper, and diagonal tracer matrix terms at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:431-457`.
The same compiled routine forms the separate temperature and salinity
contents, performs the forward recurrence, and then backward substitution at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:527-559`.
