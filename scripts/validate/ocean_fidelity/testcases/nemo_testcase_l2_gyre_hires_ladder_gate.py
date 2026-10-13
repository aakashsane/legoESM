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
    read_stage,
    read_tracer_stage12,
    read_zdf_entry,
)
from nemo_testcase_state_ulp_probe import ulp_distance  # noqa: E402

NEMO_DIMS = (126, 86, 31)
DEFAULT_ORACLES = {
    "r4": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
        "round1/oracle_gyre_r4_stage/run"
    ),
    "r4_smag": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
        "round2/oracle_gyre_r4_smag_stage/run"
    ),
}
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
    unequal = int(
        np.count_nonzero(reference[active].view(np.uint64) != value[active].view(np.uint64))
    )
    ulps = ulp_distance(reference[active], value[active])
    absolute = np.abs(reference[active] - value[active])
    worst = int(np.argmax(absolute))
    return {
        "name": name,
        "status": "AT-BAR" if unequal == 0 else "DEBT",
        "n": int(active.sum()),
        "n_unequal": unequal,
        "max_abs": float(absolute[worst]),
        "ulp_at_max_abs": int(ulps[worst]),
        "max_ulp": int(np.max(ulps)),
        "bit_bar": 0,
        "plant_one_ulp": plant_one_ulp,
    }


def run(oracle_root: Path, output: Path, *, deck: str = "r4") -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        GYRE_R4_SMAG_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree ladder certification requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if deck not in DEFAULT_ORACLES:
        raise ValueError(f"unknown ladder deck {deck!r}")
    base = GYRE_R4_NAMELIST if deck == "r4" else GYRE_R4_SMAG_NAMELIST
    case = f"GYRE-zco-{deck}"
    namelist = base._replace(nn_itend=10, nn_stock=10, nn_write=10)
    card = build_nemo_testcase_card("GYRE-zco", namelist=namelist)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    def host_and_clear(value):
        host = jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf)
            if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
            value,
        )
        jax.clear_caches()
        return host

    masks = expected_masks(card)
    state = card.recipe.initial_state
    steps = []
    controls = []
    rk_stages = []
    tracer_stage1_transport = []
    tracer_stage1_walk = []
    vertical_candidate = None
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
            reference = oracle[field] if field == "ssh" else oracle[field][..., :30]
            row = exact_score(
                f"{case}.kt{kt}.before.{field}", reference, candidate[field], masks[field]
            )
            rows.append(row)
            if kt == 1:
                planted = exact_score(
                    f"control.one_ulp.{field}",
                    reference,
                    candidate[field],
                    masks[field],
                    plant_one_ulp=True,
                )
                controls.append(
                    {
                        **planted,
                        "status": "FIRED" if planted["n_unequal"] > 0 else "MISSED",
                    }
                )
        steps.append({"kt": kt, "Nbb": oracle["Nbb"], "rows": rows})
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state_before = state
        state = model.step(state_before, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        if kt == 1:
            state = host_and_clear(state)
            for stage in (1, 2):
                stage_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_momentum_stage=stage,
                        expose_tracer_stage=stage,
                    ),
                )
                stage_state = stage_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                stage_state = host_and_clear(stage_state)
                stage_oracle = read_stage(
                    oracle_root / f"oracle_stage_kt{kt:08d}_s{stage}.bin",
                    stage,
                    dims=NEMO_DIMS,
                )
                stage_rows = []
                for field in FIELDS:
                    reference = (
                        stage_oracle[field]
                        if field == "ssh"
                        else stage_oracle[field][..., :30]
                    )
                    stage_rows.append(
                        exact_score(
                            f"{case}.kt1.rk_stage{stage}.{field}",
                            reference,
                            lego_fields(stage_state)[field],
                            masks[field],
                        )
                    )
                rk_stages.append({"stage": stage, "rows": stage_rows})
            stage_oracle = read_stage(
                oracle_root / f"oracle_stage_kt{kt:08d}_s3.bin",
                3,
                dims=NEMO_DIMS,
            )
            stage_rows = []
            for field in FIELDS:
                reference = (
                    stage_oracle[field]
                    if field == "ssh"
                    else stage_oracle[field][..., :30]
                )
                stage_rows.append(
                    exact_score(
                        f"{case}.kt1.rk_stage3.{field}",
                        reference,
                        lego_fields(state)[field],
                        masks[field],
                    )
                )
            rk_stages.append({"stage": 3, "rows": stage_rows})

            if deck == "r4":
                tracer_oracle = read_tracer_stage12(
                    oracle_root / "oracle_rktracer_operands_kt00000001_s1.bin",
                    1,
                    dims=NEMO_DIMS,
                )
                transport_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_tracer_transport_stage=1,
                    ),
                )
                transport_state = transport_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                transport_state = host_and_clear(transport_state)
                transport_values = {
                    "zFu": np.asarray(transport_state.u.data)[:, 1:, :],
                    "zFv": np.asarray(transport_state.v.data)[1:, :, :],
                    "zFw": np.asarray(transport_state.T.data),
                }
                tracer_stage1_transport = [
                    exact_score(
                        f"{case}.kt1.stage1.transport.{name}",
                        tracer_oracle[name][..., :30],
                        transport_values[name],
                        masks[{"zFu": "u", "zFv": "v", "zFw": "T"}[name]],
                    )
                    for name in ("zFu", "zFv", "zFw")
                ]
                for boundary in ("after_advection", "after_sbc"):
                    boundary_model = LatLonCGridOceanModel(
                        card.recipe.grid,
                        card.recipe.z_coord,
                        card.recipe.model_config,
                        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                            expose_tracer_stage1_boundary=boundary,
                        ),
                    )
                    boundary_state = boundary_model.step(
                        state_before,
                        card.dt_s,
                        freshwater=freshwater,
                        surface_forcing=surface,
                    )
                    boundary_state = host_and_clear(boundary_state)
                    boundary_fields = lego_fields(boundary_state)
                    tracer_stage1_walk.append({
                        "boundary": boundary,
                        "rows": [
                            exact_score(
                                f"{case}.kt1.stage1.{boundary}.{field}",
                                tracer_oracle[f"{boundary}_{field}"][..., :30],
                                boundary_fields[field],
                                masks[field],
                            )
                            for field in ("T", "S")
                        ],
                    })

            if deck == "r4":
                import jax.numpy as jnp

                zdf = read_zdf_entry(
                    oracle_root / "oracle_zdf_entry_kt00000001.bin",
                    dims=NEMO_DIMS,
                )
                seeded = model._seed_tke_preclosure_carry(state_before)
                model.prime_step_caches(seeded)
                lego_avt, lego_avm = model.diagnose_vertical_K(
                    seeded, card.dt_s, surface
                )
                lego_avt, lego_avm = host_and_clear((lego_avt, lego_avm))
                interface_mask = (
                    np.asarray(card.recipe.z_coord.is_active)[..., 1:]
                    & (np.asarray(seeded.land_mask.data) > 0.5)[..., None]
                )
                coefficient_rows = [
                    exact_score(
                        f"{case}.kt1.zdf_entry.avt",
                        zdf["avt"][..., 1:30],
                        lego_avt,
                        interface_mask,
                    ),
                    exact_score(
                        f"{case}.kt1.zdf_entry.avm",
                        zdf["avm"][..., 1:30],
                        lego_avm,
                        interface_mask,
                    ),
                ]
                vertical_control_state = model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                    _vertical_K_test_override=(
                        jnp.asarray(zdf["avt"][..., 1:30]),
                        jnp.asarray(zdf["avm"][..., 1:30]),
                    ),
                )
                vertical_control_state = host_and_clear(vertical_control_state)
                oracle2 = read_entry(
                    oracle_root / "oracle_step_entry_kt00000002.bin",
                    dims=NEMO_DIMS,
                )
                faithful = lego_fields(state)
                controlled = lego_fields(vertical_control_state)
                arm_rows = []
                for field in FIELDS:
                    reference = oracle2[field] if field == "ssh" else oracle2[field][..., :30]
                    arm_rows.append(
                        {
                            "field": field,
                            "faithful": exact_score(
                                f"{case}.kt2.faithful.{field}",
                                reference,
                                faithful[field],
                                masks[field],
                            ),
                            "oracle_zdf_override": exact_score(
                                f"{case}.kt2.oracle_zdf_override.{field}",
                                reference,
                                controlled[field],
                                masks[field],
                            ),
                        }
                    )
                vertical_candidate = {
                    "candidate": "stage-entry TKE/EVD vertical coefficients",
                    "changed_operand": "avt/avm profile bundle only",
                    "coefficient_rows": coefficient_rows,
                    "one_variable_arm": arm_rows,
                }
    passed = all(row["status"] == "AT-BAR" for step in steps for row in step["rows"]) and all(
        control["status"] == "FIRED" for control in controls
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-exact-ladder-v1",
        "worktree": worktree_stamp(),
        "case": case,
        "deck": deck,
        "oracle_root": str(oracle_root),
        "policy": "fp64/x64/cpu/libm certified",
        "acceptance": "zero unequal active float64 values in every field/step",
        "namelist": namelist._asdict(),
        "steps": steps,
        "rk_stage_walk": rk_stages,
        "tracer_stage1_transport": tracer_stage1_transport,
        "tracer_stage1_walk": tracer_stage1_walk,
        "vertical_candidate": vertical_candidate,
        "controls": controls,
        "verdict": "PASS" if passed else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=tuple(DEFAULT_ORACLES), default="r4")
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root or DEFAULT_ORACLES[args.deck], args.output, deck=args.deck)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
