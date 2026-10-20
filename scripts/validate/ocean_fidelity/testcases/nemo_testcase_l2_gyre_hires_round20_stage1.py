#!/usr/bin/env python3
"""Locate the earliest common-state GYRE momentum stage boundary.

The admitted scalar/vector NEMO records and legoESM production-JIT trace all
start from the same day-180 restart.  This gate compares the completed stage-1
RHS, external-mode handoff, and corrected stage outputs before asking for a
new source-ordered contributor record.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from nemo_testcase_l2_gyre_hires_round17_vorticity import (
    COMMON_RESTART,
    COMMON_RESTART_SHA256,
    NATIVE_DIMS,
    common_card,
    common_state,
    manifest_contract,
    sha256,
    wordwise_result,
)
from nemo_testcase_l2_gyre_phase3_gate import (
    expected_masks,
    read_stage,
    require,
)


VECTOR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_vector"
)
SCALAR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_scalar"
)
RHS_RECORD = "oracle_rhs_kt00000001.bin"
BT_RECORD = "oracle_bt_frames_kt00004321.bin"
STEP = 4321
ACTIVE_LEVELS = 30
TWIN_CLEAR_FACTOR = 4.0
FACE_ORDER = ("u", "v")


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[
        2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_completed_rhs(path: Path) -> dict[str, object]:
    """Read the stage-1 3-D RHS emitted immediately after ``stp_2D``."""
    raw = path.read_bytes()
    require(len(raw) >= 44, f"{path}: short RHS record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=7i", raw[16:44])
    keys = ("version", "kt", "Krhs", "nx", "ny", "nz", "bits")
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["nz"], header["bits"])
        == (1, STEP, *NATIVE_DIMS, 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=44)
    count = header["nx"] * header["ny"] * header["nz"]
    require(values.size == 2 * count, f"{path}: malformed RHS payload")
    fields = {
        "u": _xyz(values[:count], header["nx"], header["ny"], header["nz"]),
        "v": _xyz(values[count:], header["nx"], header["ny"], header["nz"]),
    }
    for name, field in fields.items():
        require(np.all(np.isfinite(field[..., :ACTIVE_LEVELS])),
                f"{path}: non-finite active {name} RHS word")
    return {"header": header, **fields}


def read_bt_handoff(path: Path) -> dict[str, object]:
    """Read the split-explicit output consumed by every RK stage."""
    raw = path.read_bytes()
    require(len(raw) >= 40, f"{path}: short barotropic record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=6i", raw[16:40])
    keys = ("version", "kt", "Kaa", "nx", "ny", "bits")
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L1_BTFRM_1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["bits"])
        == (1, STEP, NATIVE_DIMS[0], NATIVE_DIMS[1], 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=40)
    count = header["nx"] * header["ny"]
    require(values.size == 4 * count, f"{path}: malformed barotropic payload")
    names = ("uu_b", "vv_b", "un_adv", "vn_adv")
    fields = {
        name: _xy(values[index * count:(index + 1) * count],
                  header["nx"], header["ny"])
        for index, name in enumerate(names)
    }
    for name, field in fields.items():
        require(np.all(np.isfinite(field)),
                f"{path}: non-finite active {name} word")
    return {"header": header, **fields}


def envelope_score(
    candidate: np.ndarray,
    scalar: np.ndarray,
    vector: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    """Score exact words and the preregistered four-times twin maximum."""
    candidate = np.asarray(candidate, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    require(candidate.shape == scalar.shape == vector.shape == active.shape,
            "provider shapes differ")
    require(np.any(active), "comparison mask is empty")
    candidate_row = wordwise_result(candidate[active], scalar[active])
    twin_row = wordwise_result(scalar[active], vector[active])
    candidate_max = float(candidate_row["max_abs"])
    twin_max = float(twin_row["max_abs"])
    if twin_max == 0.0:
        ratio = None if candidate_max == 0.0 else float("inf")
        clears = candidate_max > 0.0
    else:
        ratio = candidate_max / twin_max
        clears = ratio >= TWIN_CLEAR_FACTOR
    return {
        "legoesm_minus_scalar": candidate_row,
        "scalar_minus_vector": twin_row,
        "candidate_to_twin_max_ratio": ratio,
        "twin_clear_factor": TWIN_CLEAR_FACTOR,
        "clears_four_times_twin": bool(clears),
    }


def first_clearing_stage(stages: list[dict[str, object]]) -> dict | None:
    """Return the first stage/face row clearing the registered envelope."""
    for stage in stages:
        rows = stage["rows"]
        for face in FACE_ORDER:
            if rows[face]["clears_four_times_twin"]:
                return {"stage": stage["stage"], "field": face,
                        "score": rows[face]}
    return None


def _face_fields(values: tuple) -> dict[str, np.ndarray]:
    return {
        "u": np.asarray(values[0], dtype=np.float64)[:, 1:, :],
        "v": np.asarray(values[1], dtype=np.float64)[1:, :, :],
    }


def _active_levels(field: np.ndarray) -> np.ndarray:
    """Drop NEMO's closed jpk slot; legoESM stores the 30 active levels."""
    field = np.asarray(field, dtype=np.float64)
    require(field.ndim == 3 and field.shape[-1] == NATIVE_DIMS[-1],
            f"oracle field has unexpected shape {field.shape}")
    return field[..., :ACTIVE_LEVELS]


def _plant_one_ulp(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    planted = np.array(field, dtype=np.float64, copy=True)
    index = tuple(np.argwhere(mask)[0])
    direction = np.inf if planted[index] >= 0.0 else -np.inf
    planted[index] = np.nextafter(planted[index], direction)
    return planted


def _state_bytes_equal(left, right) -> tuple[bool, int]:
    import jax

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    require(left_tree == right_tree, "trace and ordinary pytrees differ")
    mismatches = 0
    for candidate, reference in zip(left_leaves, right_leaves, strict=True):
        candidate = np.ascontiguousarray(np.asarray(candidate))
        reference = np.ascontiguousarray(np.asarray(reference))
        require(candidate.shape == reference.shape
                and candidate.dtype == reference.dtype,
                "trace and ordinary state leaf shape/dtype differs")
        mismatches += int(np.count_nonzero(
            candidate.view(np.uint8) != reference.view(np.uint8)))
    return mismatches == 0, mismatches


def run(
    vector_root: Path,
    scalar_root: Path,
    *,
    plant: str = "none",
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "round-20 gate requires CPU")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "round-20 policy is not certified fp64/libm")
    require(sha256(COMMON_RESTART) == COMMON_RESTART_SHA256,
            "common developed restart moved")
    stamp = worktree_stamp()
    require(stamp["clean"],
            f"round-20 stage walk requires clean tree: {stamp['dirty_paths']}")

    providers = {
        "vector": manifest_contract(vector_root, "vector-math-conda"),
        "scalar": manifest_contract(scalar_root, "scalar-math-conda"),
    }
    card = common_card()
    state = common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, STEP)
    trace = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    ).step(state, dt=card.dt_s, freshwater=freshwater,
           surface_forcing=surface)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
    ).step(state, dt=card.dt_s, freshwater=freshwater,
           surface_forcing=surface)
    trace = jax.device_get(trace)
    ordinary = jax.device_get(ordinary)
    passive, passive_bytes = _state_bytes_equal(trace.state_after, ordinary)
    require(passive, f"live-stage trace moved {passive_bytes} state bytes")

    masks = expected_masks(card)
    stages = []
    for stage_index in (1, 2):
        scalar = read_stage(
            scalar_root / "instrumented"
            / f"oracle_stage_kt{STEP:08d}_s{stage_index}.bin",
            stage_index, NATIVE_DIMS, expected_kt=STEP)
        vector = read_stage(
            vector_root / "instrumented"
            / f"oracle_stage_kt{STEP:08d}_s{stage_index}.bin",
            stage_index, NATIVE_DIMS, expected_kt=STEP)
        live = _face_fields(trace.stage_outputs[stage_index - 1])
        rows = {}
        for face in FACE_ORDER:
            candidate = live[face]
            if plant == f"stage{stage_index}-{face}":
                candidate = _plant_one_ulp(candidate, masks[face])
            rows[face] = envelope_score(
                candidate, _active_levels(scalar[face]),
                _active_levels(vector[face]), masks[face])
        stages.append({"stage": stage_index, "rows": rows})

    scalar_rhs = read_completed_rhs(
        scalar_root / "instrumented" / RHS_RECORD)
    vector_rhs = read_completed_rhs(
        vector_root / "instrumented" / RHS_RECORD)
    live_rhs = _face_fields(trace.stage1_full_rhs)
    rhs_rows = {}
    for face in FACE_ORDER:
        candidate = live_rhs[face]
        if plant == f"rhs-{face}":
            candidate = _plant_one_ulp(candidate, masks[face])
        rhs_rows[face] = envelope_score(
            candidate, _active_levels(scalar_rhs[face]),
            _active_levels(vector_rhs[face]), masks[face])

    scalar_bt = read_bt_handoff(
        scalar_root / "instrumented" / BT_RECORD)
    vector_bt = read_bt_handoff(
        vector_root / "instrumented" / BT_RECORD)
    live_bt = {
        "u": np.asarray(trace.barotropic_targets[0], dtype=np.float64)[:, 1:],
        "v": np.asarray(trace.barotropic_targets[1], dtype=np.float64)[1:, :],
    }
    bt_rows = {}
    for face, name in (("u", "uu_b"), ("v", "vv_b")):
        active = masks[face][..., 0]
        candidate = live_bt[face]
        if plant == f"bt-{face}":
            candidate = _plant_one_ulp(candidate, active)
        bt_rows[face] = envelope_score(
            candidate, scalar_bt[name], vector_bt[name], active)

    first_stage = first_clearing_stage(stages)
    rhs_clears = any(
        rhs_rows[face]["clears_four_times_twin"] for face in FACE_ORDER)
    bt_clears = any(
        bt_rows[face]["clears_four_times_twin"] for face in FACE_ORDER)
    if first_stage is None:
        next_boundary = "NO_STAGE_BOUNDARY_CLEARS_TWIN"
    elif first_stage["stage"] == 1 and rhs_clears:
        next_boundary = "STAGE1_SOURCE_ORDERED_RHS_ACQUISITION_REQUIRED"
    elif first_stage["stage"] == 1 and bt_clears:
        next_boundary = "STAGE1_EXTERNAL_MODE_WALK_REQUIRED"
    elif first_stage["stage"] == 1:
        next_boundary = "STAGE1_RAW_UPDATE_AND_CORRECTION_RECORD_REQUIRED"
    else:
        next_boundary = "STAGE2_SOURCE_ORDERED_RHS_ACQUISITION_REQUIRED"

    if plant != "none":
        planted_rows = []
        for stage in stages:
            planted_rows.extend(stage["rows"].values())
        planted_rows.extend(rhs_rows.values())
        planted_rows.extend(bt_rows.values())
        require(any(row["legoesm_minus_scalar"]["unequal_words"] > 0
                    for row in planted_rows),
                "active one-ULP plant did not fire")

    return {
        "format": "gyre-r4-smag-round20-stage-boundary-v1",
        "worktree": stamp,
        "common_restart": {
            "path": str(COMMON_RESTART),
            "sha256": COMMON_RESTART_SHA256,
            "step": STEP - 1,
            "next_step": STEP,
        },
        "policy": "fp64-libm-certified-cpu",
        "jax_backend": jax.default_backend(),
        "providers": providers,
        "trace_passivity": {
            "bit_exact": passive,
            "mismatched_bytes": passive_bytes,
        },
        "plant": plant,
        "stages": stages,
        "completed_stage1_rhs": rhs_rows,
        "external_mode_handoff": bt_rows,
        "first_stage_beyond_twin": first_stage,
        "next_boundary": next_boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, default=VECTOR_ROOT)
    parser.add_argument("--scalar-root", type=Path, default=SCALAR_ROOT)
    parser.add_argument("--plant", default="none", choices=(
        "none", "stage1-u", "stage1-v", "stage2-u", "stage2-v",
        "rhs-u", "rhs-v", "bt-u", "bt-v",
    ))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.vector_root, args.scalar_root, plant=args.plant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    print(json.dumps({
        "first_stage_beyond_twin": report["first_stage_beyond_twin"],
        "next_boundary": report["next_boundary"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
