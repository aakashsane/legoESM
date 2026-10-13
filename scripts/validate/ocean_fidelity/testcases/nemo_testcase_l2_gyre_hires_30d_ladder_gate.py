#!/usr/bin/env python3
"""Certified CPU exact gate for r4_smag day-boundary stage entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_hires_ladder_gate import (  # noqa: E402
    FIELDS,
    NEMO_DIMS,
    exact_score,
)
from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    expected_masks,
    lego_fields,
    read_entry,
)

DEFAULT_ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round6/oracle_gyre_r4_smag_daily_stage/run"
)
STEPS_PER_DAY = 24
NDAYS = 30


def day_boundary_schedule(ndays: int = NDAYS) -> tuple[int, ...]:
    """Return entry indices for initial state plus each completed day."""
    if ndays < 1:
        raise ValueError("ndays must be positive")
    return tuple(1 + STEPS_PER_DAY * day for day in range(ndays + 1))


def run(oracle_root: Path, output: Path, *, max_day: int = NDAYS) -> dict:
    import jax
    import numpy as np

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("30-day ladder certification requires CPU")
    if not 1 <= max_day <= NDAYS:
        raise ValueError(f"max_day must be in 1..{NDAYS}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    final_kt = day_boundary_schedule(max_day)[-1]
    namelist = GYRE_R4_SMAG_NAMELIST._replace(
        nn_itend=final_kt, nn_stock=final_kt, nn_write=STEPS_PER_DAY)
    card = build_nemo_testcase_card("GYRE-zco", namelist=namelist)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    masks = expected_masks(card)
    state = card.recipe.initial_state
    boundaries = set(day_boundary_schedule(max_day))
    rows_by_day = []
    controls = []

    for kt in range(1, final_kt + 1):
        if kt in boundaries:
            day = (kt - 1) // STEPS_PER_DAY
            path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
            oracle = read_entry(path, dims=NEMO_DIMS)
            if (oracle["kt"], oracle["Nbb"]) != (kt, 1):
                raise ValueError(f"{path}: expected odd kt with Nbb=1")
            candidate = lego_fields(state)
            rows = []
            for field in FIELDS:
                reference = (
                    oracle[field] if field == "ssh" else oracle[field][..., :30])
                rows.append(exact_score(
                    f"GYRE-zco-r4_smag.day{day}.{field}",
                    reference, candidate[field], masks[field]))
                if day == 0:
                    planted = exact_score(
                        f"control.one_ulp.{field}", reference,
                        candidate[field], masks[field], plant_one_ulp=True)
                    controls.append({
                        **planted,
                        "status": "FIRED" if planted["n_unequal"] else "MISSED",
                    })
            rows_by_day.append({"day": day, "kt": kt, "Nbb": 1, "rows": rows})
        if kt == final_kt:
            break
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state = model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        # Bound device retention during this intentionally per-step certified
        # path.  This is not the fused GPU long-run program.
        if kt % STEPS_PER_DAY == 0:
            state = jax.tree_util.tree_map(
                lambda leaf: np.asarray(leaf)
                if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
                state,
            )

    passed = (
        all(row["status"] == "AT-BAR"
            for boundary in rows_by_day for row in boundary["rows"])
        and all(control["status"] == "FIRED" for control in controls)
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-smag-30d-exact-ladder-v1",
        "worktree": worktree_stamp(),
        "case": "GYRE-zco-r4_smag",
        "oracle_root": str(oracle_root),
        "policy": "fp64/x64/cpu/libm certified; per-step loop",
        "entry_semantics": (
            "kt=1 is initial state; kt=1+24*d is the state after d complete days"
        ),
        "acceptance": "zero unequal active float64 values at all 31 boundaries",
        "namelist": namelist._asdict(),
        "boundaries": rows_by_day,
        "controls": controls,
        "verdict": "PASS" if passed else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-day", type=int, default=NDAYS)
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, args.output, max_day=args.max_day)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
