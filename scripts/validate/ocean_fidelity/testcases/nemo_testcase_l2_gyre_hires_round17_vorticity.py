#!/usr/bin/env python3
"""Common-state stage-3 ENE-q discriminator for quarter-degree GYRE.

Both NEMO arithmetic flavours and legoESM start from the same vector-math
day-180 restart.  The primary row is the round-16 western-boundary native
surface-V signed mean, narrowed to the production stage-3 vorticity/Coriolis
call.  Only the already formed NEMO q operand is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_hires_divergence import wet_span_bins  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

VECTOR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_vector"
)
SCALAR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_scalar"
)
COMMON_RESTART = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round15/oracle_gyre_r4_smag_budget_180d_corrected/control/"
    "GYRE_R4_SMAG_R14_BUDGET_00004320_restart.nc"
)
COMMON_RESTART_SHA256 = (
    "a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12"
)
STAGE_NAME = "oracle_r17_stage3_vor_kt00004321.bin"
OUTPUT_RESTART = "GYRE_R17_VORTEX_00004321_restart.nc"
NATIVE_DIMS = (126, 86, 31)
ACTIVE_LEVELS = 30
TWIN_CLEAR_FACTOR = 4.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)


def read_stage_vorticity(path: Path) -> dict[str, object]:
    """Read the RHS immediately before and after stage-3 ``dyn_vor``."""
    raw = path.read_bytes()
    require(len(raw) >= 60, f"{path}: short stage-3 record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=11i", raw[16:60])
    keys = (
        "version", "kt", "stage", "Kbb", "Kmm", "Krhs", "Kaa",
        "nx", "ny", "nz", "bits",
    )
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L2_R17VOR1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["stage"],
         header["nx"], header["ny"], header["nz"], header["bits"])
        == (1, 4321, 3, *NATIVE_DIMS, 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=60)
    n3 = header["nx"] * header["ny"] * header["nz"]
    require(values.size == 4 * n3, f"{path}: malformed payload")
    names = ("before_u", "before_v", "after_u", "after_v")
    arrays = {
        name: _xyz(values[index * n3:(index + 1) * n3],
                   header["nx"], header["ny"], header["nz"])
        for index, name in enumerate(names)
    }
    for name, array in arrays.items():
        require(np.all(np.isfinite(array[2:-2, 2:-2, :ACTIVE_LEVELS])),
                f"{path}: non-finite active {name} word")
    return {"header": header, **arrays}


def read_ene_operands(path: Path) -> dict[str, object]:
    """Read q and metric transports from one executed ``vor_ene`` call."""
    raw = path.read_bytes()
    require(len(raw) >= 48, f"{path}: short ENE operand record")
    magic = raw[:16].decode("ascii").rstrip()
    values_header = struct.unpack("=8i", raw[16:48])
    keys = ("version", "kt", "Kmm", "kvor", "nx", "ny", "nz", "bits")
    header = dict(zip(keys, values_header, strict=True))
    require(magic == "NEMO_L2_R17Q__1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["nz"], header["bits"])
        == (1, 4321, NATIVE_DIMS[0], NATIVE_DIMS[1], ACTIVE_LEVELS, 64),
        f"{path}: bad header {header}",
    )
    require(header["kvor"] == 4,
            f"{path}: expected executed np_CRV=4, got {header['kvor']}")
    values = np.frombuffer(raw, dtype=np.float64, offset=48)
    n2 = header["nx"] * header["ny"]
    require(values.size == 3 * n2 * header["nz"],
            f"{path}: malformed payload")
    slabs = values.reshape((header["nz"], 3, n2))
    arrays = {}
    for index, name in enumerate(("q", "Fu", "Fv")):
        arrays[name] = np.stack([
            slab.reshape((header["nx"], header["ny"]), order="F").T
            for slab in slabs[:, index, :]
        ], axis=-1)
    # dynvor.f90:537-573 does not define the outer work-array halo.  Admit only
    # words consumed by the executed q/transport stencil; refusing an
    # uninitialised, unread halo word would make the guard depend on stack
    # history, while accepting a non-finite consumed word would hide a failure.
    consumed = {
        "q": arrays["q"][1:-2, 1:-2],
        "Fu": arrays["Fu"][2:-1, 1:-2],
        "Fv": arrays["Fv"][1:-2, 2:-1],
    }
    for name, array in consumed.items():
        require(np.all(np.isfinite(array)),
                f"{path}: non-finite consumed {name} word")
    return {"header": header, **arrays}


def _manifest_contract(root: Path, flavour: str) -> dict[str, object]:
    """Verify the closed acquisition and same-binary passivity control."""
    manifest = root / "manifest.sha256"
    require(manifest.is_file(), f"{root}: missing manifest.sha256")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        member = Path(relative)
        require(not member.is_absolute() and ".." not in member.parts,
                f"{root}: manifest member escapes root: {relative}")
        path = root / relative
        require(path.is_file(), f"{root}: manifest member missing: {relative}")
        require(sha256(path) == digest,
                f"{root}: manifest member moved: {relative}")
        checked += 1
    require(checked > 0, f"{root}: empty manifest")
    recorded_flavour = (root / "manifest/precision_flavour.txt").read_text(
        encoding="utf-8").strip()
    require(recorded_flavour == flavour,
            f"{root}: precision flavour {recorded_flavour!r} != {flavour!r}")
    restart_line = (root / "manifest/common_restart.sha256").read_text(
        encoding="utf-8").strip()
    require(restart_line.startswith(COMMON_RESTART_SHA256 + "  "),
            f"{root}: common restart pin differs")
    control = root / "control" / OUTPUT_RESTART
    instrumented = root / "instrumented" / OUTPUT_RESTART
    require(sha256(control) == sha256(instrumented),
            f"{root}: writer moved the terminal restart")
    return {
        "root": str(root.resolve()),
        "precision_flavour": flavour,
        "producer_commit": (root / "manifest/producer_commit.txt").read_text(
            encoding="utf-8").strip(),
        "manifest_sha256": sha256(manifest),
        "verified_members": checked,
        "terminal_restart_sha256": sha256(control),
    }


def _native_v_term(record: dict[str, object]) -> np.ndarray:
    """Map NEMO's accumulated stage-3 V increment onto legoESM V faces."""
    from legoesm.ocean.fidelity.nemo_state_bridge import _v_north_to_face

    native = np.asarray(record["after_v"] - record["before_v"])
    owned = native[2:-2, 2:-2, :ACTIVE_LEVELS]
    return np.asarray(_v_north_to_face(owned))


def _q_to_legoesm(operands: dict[str, object]) -> np.ndarray:
    """Map NEMO F points, including the live south/west halo, to vertices."""
    q = np.asarray(operands["q"])
    mapped = q[1:-2, 1:-2, :]
    require(mapped.shape == (83, 123, ACTIVE_LEVELS),
            f"mapped q has wrong shape {mapped.shape}")
    return mapped


def signed_row(candidate: np.ndarray, reference: np.ndarray,
               selection: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    require(candidate.shape == reference.shape,
            "stage-3 row provider shapes differ")
    require(selection.shape == candidate.shape,
            "stage-3 row selection shape differs")
    require(np.any(selection), "western-boundary surface-V row is empty")
    require(np.all(np.isfinite(candidate[selection]))
            and np.all(np.isfinite(reference[selection])),
            "stage-3 row has non-finite active values")
    return float(np.mean(candidate[selection] - reference[selection],
                         dtype=np.float64))


def substitution_verdict(
    baseline: np.ndarray,
    substituted: np.ndarray,
    vector: np.ndarray,
    scalar: np.ndarray,
    selection: np.ndarray,
    untouched_exact: bool,
) -> dict[str, object]:
    """Apply the preregistered common-state NEMO-envelope verdict."""
    baseline_minus_vector = signed_row(baseline, vector, selection)
    substituted_minus_scalar = signed_row(substituted, scalar, selection)
    scalar_minus_vector = signed_row(scalar, vector, selection)
    envelope = abs(scalar_minus_vector)
    residual = abs(substituted_minus_scalar)
    candidate = abs(baseline_minus_vector)
    ratio = None if envelope == 0.0 else candidate / envelope
    confirms = residual <= envelope and untouched_exact
    return {
        "baseline_legoesm_minus_vector_signed_mean_m_s2": baseline_minus_vector,
        "substituted_legoesm_minus_scalar_signed_mean_m_s2":
            substituted_minus_scalar,
        "scalar_minus_vector_signed_mean_m_s2": scalar_minus_vector,
        "nemo_twin_envelope_abs_m_s2": envelope,
        "substituted_residual_abs_m_s2": residual,
        "baseline_candidate_to_twin_ratio": ratio,
        "round16_four_times_context_clears": (
            candidate > 0.0 if envelope == 0.0
            else ratio is not None and ratio >= TWIN_CLEAR_FACTOR
        ),
        "untouched_rows_bit_exact": untouched_exact,
        "verdict": "CONFIRM_Q_FORMATION" if confirms else "REFUTE_Q_FORMATION",
    }


def _load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _common_state(card):
    """Bridge the pinned restart, including the six external-mode histories."""
    import jax.numpy as jnp

    owners = _load_module("_r17_gyre_year_owners",
                          "nemo_testcase_l2_gyre_year_owners.py")
    year = _load_module("_r17_gyre_year_fromrest",
                        "nemo_testcase_l2_gyre_year_fromrest.py")
    state, _ = owners._gyre_checkpoint_state(card, COMMON_RESTART)
    payload = year._load_daily_reset_payload(COMMON_RESTART, ACTIVE_LEVELS)
    require(payload["sha256"] == COMMON_RESTART_SHA256,
            "restart bridge read a different common state")
    history = (
        jnp.asarray(year._face_history(payload["ub_e"], "u")),
        jnp.asarray(year._face_history(payload["ubb_e"], "u")),
        jnp.asarray(year._face_history(payload["vb_e"], "v")),
        jnp.asarray(year._face_history(payload["vbb_e"], "v")),
        jnp.asarray(payload["sshb_e"]),
        jnp.asarray(payload["sshbb_e"]),
    )
    return state._replace(bt_hist=history)


def _common_card():
    """Build the registered physical deck; step 4321 belongs to the forcing."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_nemo_testcase_card,
    )

    return build_nemo_testcase_card(
        "GYRE-zco", namelist=GYRE_R4_SMAG_NAMELIST)


def run(vector_root: Path, scalar_root: Path, *, plant_q: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "round-17 gate requires CPU")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "production JIT is required")
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "round-17 policy is not certified fp64/libm")
    require(sha256(COMMON_RESTART) == COMMON_RESTART_SHA256,
            "common developed restart moved")

    vector_contract = _manifest_contract(vector_root, "vector-math-conda")
    scalar_contract = _manifest_contract(scalar_root, "scalar-math-conda")
    vector_stage = read_stage_vorticity(vector_root / "instrumented" / STAGE_NAME)
    scalar_stage = read_stage_vorticity(scalar_root / "instrumented" / STAGE_NAME)
    require(vector_stage["header"] == scalar_stage["header"],
            "scalar/vector stage headers differ")
    kmm = int(vector_stage["header"]["Kmm"])
    q_name = f"oracle_r17_ene_q_kmm{kmm}_kt00004321.bin"
    vector_operands = read_ene_operands(vector_root / "instrumented" / q_name)
    scalar_operands = read_ene_operands(scalar_root / "instrumented" / q_name)
    require(vector_operands["header"] == scalar_operands["header"],
            "scalar/vector ENE headers differ")

    card = _common_card()
    state = _common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, 4321)
    scalar_q = _q_to_legoesm(scalar_operands)

    mask = np.asarray(state.v_mask.data, dtype=bool)
    western = wet_span_bins(mask) == 0
    selection = western[..., None] & np.zeros(
        (*mask.shape, ACTIVE_LEVELS), dtype=bool)
    selection[..., 0] = western
    if plant_q:
        scalar_q = np.array(scalar_q, copy=True)
        face = np.argwhere(western)[0]
        scalar_q[face[0], face[1], 0] += 1.0

    def expose(name: str, q_override=None) -> np.ndarray:
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_operator=name,
                expose_momentum_operator_stage=3,
                stage3_vorticity_q_override=q_override,
            ),
        )
        exposed = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        return np.asarray(exposed.v.data)

    baseline_hpg = expose("hpg")
    baseline_term = expose("vorticity")
    baseline_advection = expose("advection")
    q_array = jnp.asarray(scalar_q)
    substituted_hpg = expose("hpg", q_array)
    substituted_term = expose("vorticity", q_array)
    substituted_advection = expose("advection", q_array)
    untouched = {
        "stage3_after_hpg_v": bool(np.array_equal(
            baseline_hpg, substituted_hpg)),
        "stage3_advection_v": bool(np.array_equal(
            baseline_advection, substituted_advection)),
    }
    vector_term = _native_v_term(vector_stage)
    scalar_term = _native_v_term(scalar_stage)
    verdict = substitution_verdict(
        baseline_term, substituted_term, vector_term, scalar_term,
        selection, all(untouched.values()))
    if plant_q:
        require(verdict["verdict"] != "CONFIRM_Q_FORMATION",
                "planted active q change did not make the verdict refuse")

    q_selection = np.zeros_like(scalar_q, dtype=bool)
    q_selection[:, :-1, 0] |= western
    q_selection[:, 1:, 0] |= western
    q_diff = np.asarray(scalar_q) - _q_to_legoesm(vector_operands)
    report = {
        "format": "gyre-r4-smag-round17-stage3-vorticity-v1",
        "worktree": worktree_stamp(),
        "common_restart": {
            "path": str(COMMON_RESTART), "sha256": COMMON_RESTART_SHA256,
            "step": 4320, "next_step": 4321,
        },
        "policy": "fp64-libm-certified-cpu",
        "jax_backend": jax.default_backend(),
        "plant_q": plant_q,
        "providers": {"vector": vector_contract, "scalar": scalar_contract},
        "stage_header": vector_stage["header"],
        "operand_record": q_name,
        "reduction": (
            "stage-3 native surface V; western-boundary wet-span bin; "
            "signed mean"
        ),
        "nemo_scalar_minus_vector_q": {
            "active_surface_max_abs_s-1_m-1": float(np.max(
                np.abs(q_diff[q_selection]))),
            "active_surface_unequal_words": int(np.count_nonzero(
                q_diff[q_selection] != 0.0)),
        },
        "untouched_rows": untouched,
        "result": verdict,
        "source_citations": [
            "compiled stprk3_stg.f90:399-410",
            "compiled dynvor.f90:234-239",
            "compiled dynvor.f90:536-573",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, default=VECTOR_ROOT)
    parser.add_argument("--scalar-root", type=Path, default=SCALAR_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-q", action="store_true")
    args = parser.parse_args()
    report = run(args.vector_root, args.scalar_root, plant_q=args.plant_q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(
        "GYRE_R17_STAGE3_VORTICITY " + report["result"]["verdict"]
        + f" ratio={report['result']['baseline_candidate_to_twin_ratio']}"
        + f" substituted={report['result']['substituted_residual_abs_m_s2']:.17e}"
        + f" envelope={report['result']['nemo_twin_envelope_abs_m_s2']:.17e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
