# GYRE high-resolution round 20 receipt

Date: 2026-10-19. Status: **HELD FOR THE SOURCE-ORDERED NEMO
ACQUISITION -- NO PHYSICS CHANGE**. The common-state walk localises the first
oversized velocity boundary to stage 1 and proves that the completed
three-dimensional momentum RHS is already outside the scalar/vector NEMO
envelope. The existing records do not split that RHS by contributor, so this
round does not name a transcription owner.

## Finding -> disposition -> evidence

Scalar-math NEMO, vector-math NEMO, and legoESM start from the same day-180
`r4_smag` restart (SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`)
and execute step 4321. legoESM uses the certified CPU fp64/libm policy and its
ordinary production JIT. The private live-stage trace is passive: its complete
output state and an independently compiled ordinary step are byte-identical,
with zero mismatched bytes.

The preregistered statistic is the maximum absolute residual over every wet
native U or V word. A boundary clears the inherited round-16 envelope at four
times the scalar-minus-vector NEMO maximum; exactness remains zero unequal
words. The corrected post-stage states score as follows.

| boundary | field | legoESM - scalar maximum | unequal words | scalar - vector maximum | ratio to twin |
|---|---|---:|---:|---:|---:|
| stage 1 | U | `6.202859428316998e-7 m s-1` | 285,600 / 285,600 | `1.1102230246251565e-16` | `5.587039082e9` |
| stage 1 | V | `5.890017569232242e-7 m s-1` | 284,400 / 284,400 | `1.1102230246251565e-16` | `5.305256186e9` |
| stage 2 | U | `2.711170197025603e-8 m s-1` | 285,600 / 285,600 | `2.9974720622272244e-14` | `9.0448556008e5` |
| stage 2 | V | `2.4075361593212108e-8 m s-1` | 284,400 / 284,400 | `2.997515430314124e-14` | `8.0317723638e5` |

Stage 1 U is therefore the first stage and field beyond the measured NEMO
arithmetic envelope. This supersedes the round-19 description of stage-2 Kmm
velocity as the next boundary; that state is already downstream of the larger
stage-1 separation.

The discriminator then scores two earlier boundaries.

| boundary | field | legoESM - scalar maximum | unequal words | scalar - vector maximum | ratio to twin |
|---|---|---:|---:|---:|---:|
| completed stage-1 3-D RHS | U | `5.139395332331717e-10 m s-2` | 285,600 / 285,600 | `0` | infinity |
| completed stage-1 3-D RHS | V | `4.868790573218769e-10 m s-2` | 284,400 / 284,400 | `0` | infinity |
| external-mode handoff | U | `1.4407702836338743e-8 m s-1` | 9,520 / 9,520 | `2.7755575615628914e-17` | `5.19092201e8` |
| external-mode handoff | V | `7.966726858071871e-9 m s-1` | 9,480 / 9,480 | `2.7755575615628914e-17` | `2.87031584875e8` |

The completed three-dimensional RHS clears the envelope before the external
solve or RK update. **Disposition:** walk the source contributors first. The
external-mode difference is measured debt but is not yet eligible to own the
first stage boundary. The admitted compiled program forms the shared stage-1
RHS in the order HPG, lateral diffusion, vorticity/Coriolis, kinetic-energy
gradient, and vertical advection at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stp2d.f90:136-163`, then calls the
split-explicit solver at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stp2d.f90:266-267`.

The stage-1 velocity update itself is at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:482-485`, followed by
the all-stage depth-mean correction at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:524-549`.
Vertical mixing cannot be the first stage-1 contributor because the executed
branch calls it only for stage 3 at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:521`. The RK3 driver
calls stages 1 and 2 with pointer swaps, then stage 3, at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3.f90:201-215`; there is no
Asselin/filter update to score in this RK3 sequence.

The evidence artifact is
`gyrehr_rounds/round20/stage_boundary.json`, SHA-256
`6376e9ee5e704bfcacac97f2a61c7486dd60ed1ed56a48967cec43ffca94b603`,
produced from clean commit `1b762ee67d59aba04762142e8932eaa86dbf0eda`.
It admits 127 manifested members from each round-17 provider and records their
manifest SHA-256 values as
`6efb48c02a9626e6bb268eacf1c5be028b94c3ec4a422c7717481e3aabda340e`
(scalar) and
`1586e831b980b8d43b96bdd1e13c67faa597d9c245d48c2d3ec4dcf985a8281e`
(vector).

## Acquisition and proving test

The missing evidence is one cumulative stage-1 RHS record for each NEMO math
flavour. The committed writer records the exact shared `Krhs` immediately
after each of the five calls above. It is environment-gated, single-domain,
fp64-only, refuses any non-vector-form dynamics dispatch, writes ten native
126 x 86 x 31 arrays, and never feeds a recorded value back into NEMO. Each
operator run executes the same built binary with the writer disabled and
enabled and requires byte-identical terminal restarts. The control must emit
no round-20 file; the instrumented record must be exactly 26,873,368 bytes.

Both operator preflights pass at commit `1b762ee67`: the pinned engine patch
applies with zero fuzz, the exact `r4_smag` deck resolves, and the vector and
scalar modified sources pass the Fortran syntax compiler. The tool uses
`makenemo -d 'OCE'`, registers the cloned configuration, copies the prepared
inputs, defaults to no run mode, requires `RANKS=1`, creates the evidence
parent before its space check, and uses `grep`, not an interactive-shell
`rg` function.

After both acquisitions, the already-committed scorer compares the NEMO
cumulative arrays with legoESM's already-materialised production accumulator
barriers in the same order. The first cumulative U/V row clearing four times
the same-field scalar/vector maximum selects exactly one contributor for the
next factor walk. No cumulative difference alone names a transcription owner:
the next round must substitute one independently recorded factor and require
the contributor plus downstream stage row to fall inside the NEMO envelope
without moving untouched rows.

The one-ULP control constructs a candidate from the scalar oracle, moves the
first active word by one representable binary64 value, and requires exactly
one unequal word. Header, array-size, truncation, non-finite-active-word,
paired-provider, source-order, single-rank, passivity, and five-frame layout
refusals are covered by the scorer/tool tests.

## Inherited pins / OPEN

Inherited unchanged: shipped one-degree digest `56c5171d43504b54`, its
no-worsening ceiling, 31 levels, WS-RK3, split-explicit free surface, TKE/EVD,
Redi tracers, the `r4_smag` deck, 3,600-s step, literal 360-day forcing clock,
scalar-libm certified CPU policy, vector-math climate reference, common
day-180 restart, round-16 four-times twin discriminator, the year-5 climate
PASS against both NEMO builds, and every earlier retraction. No deck,
prognostic arithmetic, shared physical kernel, climate run, or NEMO engine
source changed.

OPEN, in order:

1. Run the committed vector and scalar one-step writer acquisitions and score
   the cumulative stage-1 rows.
2. At the first measured contributor, record and substitute its source factors
   one at a time; only a single-factor collapse can name an owner.
3. Preserve the round-18 and round-19 HELD labels until the required independent
   reviews are available; round 20 does not promote either claim.
4. Post the reviewed result to issue #1455 when network access is available.

No five-year run is warranted. The earliest boundary is now measured, but the
physics owner is still UNMEASURED and the shared path is unchanged.

Choice audit -- **ASKED**: compare stages 1 and 2 at the common restart, score
each available RHS/external/update boundary against the scalar/vector NEMO
envelope, walk the first contributor, and request only missing records.
**UNASKED**: no physics edit, deck change, threshold change, long run, or
operator acquisition was performed. No user decision is needed.

## Tests and review

The focused scorer/reader/acquisition contract tests, the inherited round-17
reader and trace tests, time-level registry tests, source-citation self-tests,
and split CPU/x64 lane battery are recorded below after the final clean commit.
Both scalar/vector operator preflights report `PREFLIGHT_OK`. `git diff
--check` passes.

Review status is **HELD** pending the campaign's two independent reviewers.
This round lands no physics claim: it reports a measured boundary and requests
the record required to continue.

## Compiled-source citations

The compiled-source claims above are bound to the exact scalar writer program
that produced the admitted common-state records. The vector writer has the
same source program; its manifest differs only in the declared math flavour
and compiled binary.
