#!/usr/bin/env python3
"""Exact ten-step CPU gate for the quarter-degree GYRE stage ladder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    expected_masks,
    lego_fields,
    read_entry,
)

CASE = "GYRE-zco-r4"
NEMO_DIMS = (126, 86, 31)
DEFAULT_ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round1/oracle_gyre_r4_stage/run"
)
FIELDS = ("T", "S", "u", "v", "ssh")


def exact_score(name: str, oracle, candidate, mask, *, plant_one_ulp=False) -> dict:
    """Score the preregistered bit bar and optionally plant exactly one ULP."""
    reference = np.asarray(oracle, dtype=np.float64)
    value = np.asarray(candidate, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if reference.shape != value.shape or value.shape != active.shape:
        raise ValueError(f"{name}: shape mismatch")
    if not active.any():
        raise ValueError(f"{name}: empty active mask")
    if plant_one_ulp:
        value = value.copy()
        index = tuple(np.argwhere(active)[0])
        toward = np.inf if value[index] >= 0.0 else -np.inf
        value[index] = np.nextafter(value[index], toward)
    unequal = int(np.count_nonzero(
        reference[active].view(np.uint64) != value[active].view(np.uint64)))
    return {
        "name": name,
        "status": "AT-BAR" if unequal == 0 else "DEBT",
        "n": int(active.sum()),
        "n_unequal": unequal,
        "max_abs": float(np.max(np.abs(reference[active] - value[active]))),
        "bit_bar": 0,
        "plant_one_ulp": plant_one_ulp,
    }


def run(oracle_root: Path, output: Path) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree ladder certification requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    namelist = GYRE_R4_NAMELIST._replace(
        nn_itend=10, nn_stock=10, nn_write=10)
    card = build_nemo_testcase_card("GYRE-zco", namelist=namelist)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    masks = expected_masks(card)
    state = card.recipe.initial_state
    steps = []
    controls = []
    for kt in range(1, 11):
        path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
        oracle = read_entry(path, dims=NEMO_DIMS)
        expected_nbb = 1 if kt % 2 else 3
        if (oracle["kt"], oracle["Nbb"]) != (kt, expected_nbb):
            raise ValueError(f"{path}: wrong kt/Nbb")
        candidate = lego_fields(state)
        rows = []
        for field in FIELDS:
            reference = (
                oracle[field] if field == "ssh" else oracle[field][..., :30])
            row = exact_score(
                f"{CASE}.kt{kt}.before.{field}", reference,
                candidate[field], masks[field])
            rows.append(row)
            if kt == 1:
                planted = exact_score(
                    f"control.one_ulp.{field}", reference,
                    candidate[field], masks[field], plant_one_ulp=True)
                controls.append({
                    **planted,
                    "status": "FIRED" if planted["n_unequal"] > 0 else "MISSED",
                })
        steps.append({"kt": kt, "Nbb": oracle["Nbb"], "rows": rows})
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state = model.step(
            state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
    passed = (
        all(row["status"] == "AT-BAR" for step in steps for row in step["rows"])
        and all(control["status"] == "FIRED" for control in controls)
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-exact-ladder-v1",
        "worktree": worktree_stamp(),
        "case": CASE,
        "oracle_root": str(oracle_root),
        "policy": "fp64/x64/cpu/libm certified",
        "acceptance": "zero unequal active float64 values in every field/step",
        "namelist": namelist._asdict(),
        "steps": steps,
        "controls": controls,
        "verdict": "PASS" if passed else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
