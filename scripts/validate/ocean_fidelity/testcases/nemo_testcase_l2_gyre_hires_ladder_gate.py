#!/usr/bin/env python3
"""Exact ten-step CPU gate for the quarter-degree GYRE stage ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    _trace_native,
    bt_frame,
    expected_masks,
    lego_fields,
    read_entry,
    read_stage,
    read_bt_substeps,
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
DEFAULT_AHM_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round4/oracle_gyre_r4_smag_ahm/run"
)
ADMISSIONS_PATH = HERE / "gyre_hires_oracle_admissions.json"
ORACLE_FLAVOURS = ("scalar", "vector")
FIELDS = ("T", "S", "u", "v", "ssh")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit_oracle(
    deck: str,
    math_flavour: str,
    oracle_root: Path | None = None,
    *,
    admissions_path: Path = ADMISSIONS_PATH,
) -> dict:
    """Fail closed unless the requested oracle matches its registered bytes."""
    if math_flavour not in ORACLE_FLAVOURS:
        raise ValueError(f"unknown oracle math flavour {math_flavour!r}")
    try:
        registry = json.loads(admissions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read oracle admission registry: {exc}") from exc
    if registry.get("format") != "gyre-hires-oracle-admissions-v1":
        raise ValueError("unsupported oracle admission registry format")
    key = f"{deck}:{math_flavour}"
    try:
        registered = registry["admissions"][key]
    except KeyError as exc:
        raise ValueError(f"unregistered oracle selection {key!r}") from exc
    if (registered.get("deck"), registered.get("math_flavour")) != (
        deck, math_flavour,
    ):
        raise ValueError(f"{key}: admission deck/flavour mismatch")
    root = Path(registered["root"])
    if oracle_root is not None and oracle_root.resolve() != root.resolve():
        raise ValueError(
            f"{key}: unregistered oracle root {oracle_root}; expected {root}"
        )
    manifest = Path(registered["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"ACQUISITION_NEEDED: {manifest}")
    if _sha256(manifest) != registered["manifest_sha256"]:
        raise ValueError(f"{key}: manifest digest mismatch")
    checked = {}
    for relative, expected in registered["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{key}: digest mismatch for {relative}: {actual} != {expected}"
            )
        checked[relative] = actual
    return {
        **registered,
        "key": key,
        "root": str(root),
        "manifest": str(manifest),
        "checked_files": checked,
    }


def compiled_usrdef_math_program(
    executable: Path, *, expected_flavour: str = "vector"
) -> dict:
    """Inventory transcendental calls in the admitted usrdef_sbc routine."""
    if not executable.is_file():
        raise FileNotFoundError(f"missing admitted NEMO executable: {executable}")
    completed = subprocess.run(
        ["objdump", "-d", "--demangle",
         "--disassemble=__usrdef_sbc_MOD_usrdef_sbc_oce", str(executable)],
        check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise ValueError(f"objdump failed for {executable}: {completed.stderr.strip()}")
    calls = sorted(set(re.findall(
        r"callq?\s+[^<]*<([^>@]+)(?:@[^>]*)?>", completed.stdout)))
    transcendental_names = ("sin", "cos", "exp")
    vector_transcendentals = [
        name for name in calls
        if "_ZGV" in name and name.endswith(transcendental_names)
    ]
    scalar_transcendentals = [
        name for name in calls if name in transcendental_names
    ]
    vector_sin = [name for name in vector_transcendentals if name.endswith("sin")]
    scalar_sin = [name for name in scalar_transcendentals if name == "sin"]
    if expected_flavour == "vector":
        if ({name.rsplit("_", 1)[-1] for name in vector_transcendentals}
                != {"sin", "cos"}
                or set(scalar_transcendentals) != {"sin", "cos"}):
            raise ValueError(
                "vector oracle usrdef_sbc did not expose vector and scalar "
                "sin/cos with no exp"
            )
    elif expected_flavour == "scalar":
        if vector_transcendentals or set(scalar_transcendentals) != {"sin", "cos"}:
            raise ValueError(
                "scalar oracle usrdef_sbc did not expose scalar-only sin/cos "
                "with no exp"
            )
    else:
        raise ValueError(f"unknown expected math flavour {expected_flavour!r}")
    return {
        "executable": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "routine": "__usrdef_sbc_MOD_usrdef_sbc_oce",
        "transcendental_calls": calls,
        "vector_transcendental_calls": vector_transcendentals,
        "scalar_transcendental_calls": scalar_transcendentals,
        "vector_sin_calls": vector_sin,
        "scalar_sin_calls": scalar_sin,
        "math_flavour": expected_flavour,
    }


def wind_expression_arms(r1_rho0, stress, current_reciprocal, depth_ref, mask):
    """Replay the compiled ``stp2d`` wind expression and two discriminators."""
    rho = np.float64(r1_rho0)
    tau = np.asarray(stress, dtype=np.float64)
    live_reciprocal = np.asarray(current_reciprocal, dtype=np.float64)
    depth = np.asarray(depth_ref, dtype=np.float64)
    wet = np.asarray(mask, dtype=np.float64)
    if not (tau.shape == live_reciprocal.shape == depth.shape == wet.shape):
        raise ValueError("wind-expression operand shape mismatch")
    one = np.float64(1.0)
    stored_reciprocal = wet / ((depth + one) - wet)
    return {
        # GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stp2d.f90:198-199.
        "nemo_written": (rho * tau) * stored_reciprocal,
        "right_associated": rho * (tau * stored_reciprocal),
        "reciprocal_first": (rho * stored_reciprocal) * tau,
        "current_reciprocal": (rho * tau) * live_reciprocal,
        "stored_reciprocal": stored_reciprocal,
    }


def coefficient_branch_census(diagnostics, t_active, f_active):
    """Classify the final max/min arm on active T and F coefficient points."""
    result = {}
    for point, active in (("T", t_active), ("F", f_active)):
        wet = np.asarray(active, dtype=bool)
        lower = np.asarray(diagnostics[f"{point.lower()}_lower_clamped"], dtype=bool)
        upper = np.asarray(diagnostics[f"{point.lower()}_upper_clamped"], dtype=bool)
        if wet.shape != lower.shape or lower.shape != upper.shape or not wet.any():
            raise ValueError(f"{point}: invalid Smagorinsky branch mask")
        # The source evaluates MIN(MAX(deformation, lower), upper), so an
        # upper clamp owns a cell even if the intermediate MAX chose lower.
        arms = {
            "lower_limit": wet & lower & ~upper,
            "deformation": wet & ~lower & ~upper,
            "upper_limit": wet & upper,
        }
        count = int(np.count_nonzero(wet))
        counts = {name: int(np.count_nonzero(mask)) for name, mask in arms.items()}
        if sum(counts.values()) != count:
            raise ValueError(f"{point}: Smagorinsky branch census is not exhaustive")
        result[point] = {
            "active_count": count,
            "counts": counts,
            "fractions": {name: value / count for name, value in counts.items()},
        }
    return result


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
    active_indices = np.argwhere(active)
    unequal_bits = (
        reference[active].view(np.uint64) != value[active].view(np.uint64)
    )
    first_unequal = (
        active_indices[int(np.flatnonzero(unequal_bits)[0])]
        if unequal else None
    )
    return {
        "name": name,
        "status": "AT-BAR" if unequal == 0 else "DEBT",
        "n": int(active.sum()),
        "n_unequal": unequal,
        "max_abs": float(absolute[worst]),
        "ulp_at_max_abs": int(ulps[worst]),
        "max_ulp": int(np.max(ulps)),
        "max_abs_index": [int(index) for index in active_indices[worst]],
        "first_unequal_index": (
            [int(index) for index in first_unequal]
            if first_unequal is not None else None
        ),
        "first_unequal_ulp": (
            int(ulps[int(np.flatnonzero(unequal_bits)[0])]) if unequal else 0
        ),
        "bit_bar": 0,
        "plant_one_ulp": plant_one_ulp,
    }


def accumulated_component_score(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
    component: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Score a component at the accumulator boundary that consumed it.

    NEMO records the accumulator before and after an operator, not the
    operator addend. Floating-point endpoint subtraction is not the inverse
    of the executed addition, so start from NEMO's exact preceding
    accumulator and add the isolated legoESM component instead.
    """
    return exact_score(name, after, before + component, mask)


def _full_xyz(values: np.ndarray, dims=NEMO_DIMS) -> np.ndarray:
    nx, ny, nz = dims
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_tracer_stage3_hires(path: Path, *, dims=NEMO_DIMS) -> dict:
    """Read the P3 stage-3 tracer accumulator at the selected resolution."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKTR3_1":
        raise ValueError(f"{path}: bad stage-3 tracer magic")
    if ((version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
            != (1, 1, 3, 1, 2, 3, 3, *dims, 64)):
        raise ValueError(f"{path}: bad stage-3 tracer header")
    n3, n2 = nx * ny * nz, nx * ny
    if values.size != 16 * n3 + 3 * n2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad stage-3 tracer payload")
    names = (
        "zero_T", "zero_S", "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
        "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
        "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result = {
        name: _full_xyz(values[index * n3:(index + 1) * n3], dims)
        for index, name in enumerate(names)
    }
    offset = 16 * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        block = values[offset + index * n2:offset + (index + 1) * n2]
        result[name] = block.reshape((nx, ny), order="F")[2:-2, 2:-2].T
    result.update({"Kbb": kbb, "Kmm": kmm, "Krhs": krhs, "Kaa": kaa})
    return result


def _native_f_to_vertex(native: np.ndarray) -> np.ndarray:
    """Map NEMO F(i,j) to legoESM vertex[j+1,(i+1) mod nx]."""
    ny, nx, nz = native.shape
    result = np.zeros((ny + 1, nx + 1, nz), dtype=np.float64)
    columns = (np.arange(nx + 1) - 1) % nx
    result[1:] = native[:, columns]
    return result


def read_ahm(path: Path, *, expected_kt: int, dims=NEMO_DIMS) -> dict:
    """Read the acquired post-ldf_dyn mode-32 coefficient pair."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii")
        version, kt, kbb, nx, ny, nz, bits = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    if magic != "NEMO_L4_AHMTF__1":
        raise ValueError(f"{path}: bad AHM magic")
    if (version, kt, nx, ny, nz, bits) != (1, expected_kt, *dims, 64):
        raise ValueError(f"{path}: bad AHM header")
    count = nx * ny * nz
    if values.size != 2 * count or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad AHM payload")
    return {
        "kt": kt,
        "Kbb": kbb,
        "ahmt": _full_xyz(values[:count], dims)[..., :nz - 1],
        "ahmf": _native_f_to_vertex(
            _full_xyz(values[count:], dims))[..., :nz - 1],
    }


def read_dynldf(path: Path, *, expected_kt: int, dims=NEMO_DIMS) -> dict:
    """Read NEMO's stage-3 RHS immediately before and after dyn_ldf."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii")
        header = struct.unpack("=10i", handle.read(40))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, nx, ny, nz, bits = header
    if magic != "NEMO_L4_DYNLDF_1":
        raise ValueError(f"{path}: bad dyn_ldf magic")
    if ((version, kt, stage, nx, ny, nz, bits)
            != (1, expected_kt, 3, *dims, 64)):
        raise ValueError(f"{path}: bad dyn_ldf header")
    count = nx * ny * nz
    if values.size != 4 * count or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad dyn_ldf payload")
    blocks = [
        _full_xyz(block, dims)[..., :nz - 1]
        for block in np.split(values, 4)
    ]
    return {
        "kt": kt, "stage": stage, "Kbb": kbb, "Kmm": kmm, "Krhs": krhs,
        "u_before": blocks[0], "v_before": blocks[1],
        "u_after": blocks[2], "v_after": blocks[3],
    }


def read_stage2_terms(path: Path, *, dims=NEMO_DIMS) -> dict:
    """Read the scalar writer's source-ordered stage-2 momentum RHS."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, krhs, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKTRM_1":
        raise ValueError(f"{path}: bad stage-2 term magic")
    if ((version, kt, stage, kmm, krhs, nx, ny, nz, bits)
            != (1, 1, 2, 3, 2, *dims, 64)):
        raise ValueError(f"{path}: bad stage-2 term header")
    count = nx * ny * nz
    if values.size != 8 * count or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad stage-2 term payload")
    names = (
        "before_u", "before_v", "after_hpg_u", "after_hpg_v",
        "after_vorticity_u", "after_vorticity_v",
        "after_advection_u", "after_advection_v",
    )
    result = {
        name: _full_xyz(values[index * count:(index + 1) * count], dims)
        for index, name in enumerate(names)
    }
    result.update({"stage": stage, "Kmm": kmm, "Krhs": krhs})
    return result


def read_stage2_composition(path: Path, *, dims=NEMO_DIMS) -> dict:
    """Read raw update and post-barotropic-replacement stage-2 states."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKSTG_1":
        raise ValueError(f"{path}: bad stage-2 composition magic")
    if ((version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
            != (1, 1, 2, 1, 3, 2, 2, *dims, 64)):
        raise ValueError(f"{path}: bad stage-2 composition header")
    n3, n2 = nx * ny * nz, nx * ny
    ni = (nx - 4) * (ny - 4)
    if values.size != 10 * n3 + 2 * n2 + 2 * ni:
        raise ValueError(f"{path}: bad stage-2 composition payload")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite stage-2 composition payload")
    result = {}
    offset = 0
    for name in (
        "Kbb_u", "Kbb_v", "Kmm_u", "Kmm_v", "Krhs_u", "Krhs_v",
        "raw_Kaa_u", "raw_Kaa_v",
    ):
        result[name] = _full_xyz(values[offset:offset + n3], dims)
        offset += n3
    offset += 2 * n2  # barotropic Kaa u/v, not scored by this discriminator
    offset += 2 * ni  # zub/zvb, already represented by the post state
    result["post_mean_Kaa_u"] = _full_xyz(values[offset:offset + n3], dims)
    offset += n3
    result["post_mean_Kaa_v"] = _full_xyz(values[offset:offset + n3], dims)
    offset += n3
    if offset != values.size:
        raise ValueError(f"{path}: unread stage-2 composition payload")
    return result


def _redundant_faces(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map native NEMO east/north faces to legoESM redundancy."""
    u = np.asarray(u)[..., :30]
    v = np.asarray(v)[..., :30]
    return (
        np.concatenate([u[:, -1:, :], u], axis=1),
        np.concatenate([v[-1:, :, :], v], axis=0),
    )


def mismatch_map(reference, candidate, regions: dict[str, np.ndarray], clamps) -> dict:
    """Count bit differences by preregistered spatial/clamp class."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    unequal = reference.view(np.uint64) != candidate.view(np.uint64)
    mapped = {name: int(np.count_nonzero(unequal & mask))
              for name, mask in regions.items()}
    mapped.update({
        "lower_clamped": int(np.count_nonzero(unequal & np.asarray(clamps[0]))),
        "upper_clamped": int(np.count_nonzero(unequal & np.asarray(clamps[1]))),
    })
    mapped["all"] = int(np.count_nonzero(unequal))
    return mapped


def run(
    oracle_root: Path, output: Path, *, deck: str = "r4",
    ahm_root: Path = DEFAULT_AHM_ROOT,
    math_flavour: str = "vector",
) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask,
        nemo_smagorinsky_bilaplacian_coefficients_cgrid,
    )
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
    admission = admit_oracle(deck, math_flavour, oracle_root)
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
    tracer_stage1_transport_discriminator = None
    stage2_momentum_term_walk = None
    vertical_candidate = None
    barotropic_resolution_walk = None
    smagorinsky_coefficients = []
    smagorinsky_branch_census = []
    smagorinsky_operator = []
    ahm_records = {}
    if deck == "r4_smag":
        cell3 = (
            np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
            * np.asarray(state.land_mask.data)[..., None]
        )
        coefficient_vertex_mask = jax.vmap(
            lambda level: compute_vertex_mask(level, grid=card.recipe.grid),
            in_axes=-1, out_axes=-1,
        )(jnp.asarray(cell3))
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
        if (deck == "r4_smag" and math_flavour == "vector"
                and kt in (2, 3, 10)):
            # Rule 12: score the changed coefficient on NEMO's own Kbb
            # velocity input, not legoESM's already-diverged trajectory.
            u_input = np.concatenate(
                [np.zeros((oracle["u"].shape[0], 1, 30)), oracle["u"][..., :30]],
                axis=1,
            )
            v_input = np.concatenate(
                [np.zeros((1, oracle["v"].shape[1], 30)), oracle["v"][..., :30]],
                axis=0,
            )
            got_t, got_f, diagnostics = (
                nemo_smagorinsky_bilaplacian_coefficients_cgrid(
                    jnp.asarray(u_input), jnp.asarray(v_input), card.recipe.grid,
                    3.5, card.dt_s, mask=state.land_mask.data,
                    cell_mask=jnp.asarray(cell3), u_mask=state.u_mask.data,
                    v_mask=state.v_mask.data,
                    vertex_mask=coefficient_vertex_mask,
                    return_diagnostics=True,
                )
            )
            got_t, got_f, diagnostics = host_and_clear(
                (got_t, got_f, diagnostics))
            vertex_mask_host = np.asarray(coefficient_vertex_mask, dtype=bool)
            smagorinsky_branch_census.append({
                "kt": kt,
                "input_time_level": "Nbb before velocity",
                "active_definition": (
                    "wet T cells use cell_mask; F cells use the mode-32 "
                    "nonzero coefficient support, including free-slip walls"
                ),
                "branches": coefficient_branch_census(
                    diagnostics,
                    np.asarray(cell3, dtype=bool),
                    np.asarray(got_f != 0.0, dtype=bool),
                ),
            })
            if kt == 10:
                continue
            ahm_path = ahm_root / f"oracle_ahm_kt{kt:08d}.bin"
            if not ahm_path.is_file():
                raise FileNotFoundError(f"ACQUISITION_NEEDED: {ahm_path}")
            ahm = read_ahm(ahm_path, expected_kt=kt)
            ahm_records[kt] = ahm
            full_t = np.ones_like(ahm["ahmt"], dtype=bool)
            full_f = np.ones_like(ahm["ahmf"], dtype=bool)
            spatial = np.any(ahm["ahmf"] != 0.0, axis=-1)
            coords = np.argwhere(spatial)
            if not coords.size:
                raise ValueError(f"{ahm_path}: empty F coefficient support")
            j0, i0 = coords.min(axis=0)
            j1, i1 = coords.max(axis=0)
            edge2 = spatial & (
                (np.indices(spatial.shape)[0] == j0)
                | (np.indices(spatial.shape)[0] == j1)
                | (np.indices(spatial.shape)[1] == i0)
                | (np.indices(spatial.shape)[1] == i1)
            )
            corner2 = spatial & (
                ((np.indices(spatial.shape)[0] == j0)
                 | (np.indices(spatial.shape)[0] == j1))
                & ((np.indices(spatial.shape)[1] == i0)
                   | (np.indices(spatial.shape)[1] == i1))
            )
            regions = {
                "interior": np.broadcast_to((spatial & ~edge2)[..., None], full_f.shape),
                "walls": np.broadcast_to((edge2 & ~corner2)[..., None], full_f.shape),
                "corners": np.broadcast_to(corner2[..., None], full_f.shape),
                "outside": np.broadcast_to((~spatial)[..., None], full_f.shape),
            }
            vertex_mask_float = np.asarray(coefficient_vertex_mask, dtype=np.float64)
            legacy_f = got_f * vertex_mask_float
            smagorinsky_coefficients.append({
                "kt": kt,
                "Kbb": ahm["Kbb"],
                "NEMO_source": str(ahm_path),
                "production": {
                    "ahmt": exact_score(
                        f"{case}.kt{kt}.ahmt", ahm["ahmt"], got_t, full_t),
                    "ahmf": exact_score(
                        f"{case}.kt{kt}.ahmf", ahm["ahmf"], got_f, full_f),
                    "ahmf_mismatch_map": mismatch_map(
                        ahm["ahmf"], got_f, regions,
                        (diagnostics["f_lower_clamped"],
                         diagnostics["f_upper_clamped"]),
                    ),
                },
                "planted_old_fmask_defect": {
                    "ahmf": exact_score(
                        f"control.kt{kt}.legacy_fmask.ahmf",
                        ahm["ahmf"], legacy_f, full_f),
                    "ahmf_mismatch_map": mismatch_map(
                        ahm["ahmf"], legacy_f, regions,
                        (diagnostics["f_lower_clamped"],
                         diagnostics["f_upper_clamped"]),
                    ),
                },
                "support": {
                    "NEMO_nonzero_T": int(np.count_nonzero(ahm["ahmt"])),
                    "NEMO_nonzero_F": int(np.count_nonzero(ahm["ahmf"])),
                    "binary_vertex_mask_nonzero": int(
                        np.count_nonzero(vertex_mask_host)),
                },
            })
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state_before = state
        state = model.step(state_before, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        if (deck == "r4_smag" and math_flavour == "vector"
                and kt in (2, 3)):
            dyn_path = ahm_root / f"oracle_dynldf_kt{kt:08d}.bin"
            if not dyn_path.is_file():
                raise FileNotFoundError(f"ACQUISITION_NEEDED: {dyn_path}")
            dyn = read_dynldf(dyn_path, expected_kt=kt)

            def exposed_stage3_rhs(kind, coefficient_override=None):
                diagnostic_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_stage3_momentum_rhs=kind,
                        smagorinsky_coefficient_override=coefficient_override,
                    ),
                )
                diagnostic = diagnostic_model.step(
                    state_before, card.dt_s,
                    freshwater=freshwater, surface_forcing=surface,
                )
                return host_and_clear(diagnostic)

            ahm = ahm_records[kt]
            nemo_pair = (jnp.asarray(ahm["ahmt"]), jnp.asarray(ahm["ahmf"]))
            legacy_pair = (
                nemo_pair[0],
                nemo_pair[1] * coefficient_vertex_mask,
            )
            pre = exposed_stage3_rhs("pre_ldf")
            post_nemo = exposed_stage3_rhs("post_ldf", nemo_pair)
            post_legacy = exposed_stage3_rhs("post_ldf", legacy_pair)
            oracle_delta = {
                "u": dyn["u_after"] - dyn["u_before"],
                "v": dyn["v_after"] - dyn["v_before"],
            }
            nemo_delta = {
                "u": lego_fields(post_nemo)["u"] - lego_fields(pre)["u"],
                "v": lego_fields(post_nemo)["v"] - lego_fields(pre)["v"],
            }
            legacy_delta = {
                "u": lego_fields(post_legacy)["u"] - lego_fields(pre)["u"],
                "v": lego_fields(post_legacy)["v"] - lego_fields(pre)["v"],
            }
            substituted_rows = [
                exact_score(
                    f"{case}.kt{kt}.stage3.dynldf.NEMO_coefficient.{field}",
                    oracle_delta[field], nemo_delta[field], masks[field],
                )
                for field in ("u", "v")
            ]
            legacy_rows = [
                exact_score(
                    f"control.kt{kt}.stage3.dynldf.legacy_fmask.{field}",
                    oracle_delta[field], legacy_delta[field], masks[field],
                )
                for field in ("u", "v")
            ]
            substituted_max = max(row["max_abs"] for row in substituted_rows)
            legacy_max = max(row["max_abs"] for row in legacy_rows)
            smagorinsky_operator.append({
                "kt": kt,
                "NEMO_source": str(dyn_path),
                "stage": dyn["stage"],
                "Kbb": dyn["Kbb"],
                "Kmm": dyn["Kmm"],
                "Krhs": dyn["Krhs"],
                "changed_operand": (
                    "stage-3 ahmt/ahmf only; the stage-1 production call is unchanged"
                ),
                "NEMO_coefficient_override": substituted_rows,
                "planted_old_fmask_override": legacy_rows,
                "owner": (
                    "COEFFICIENT: erroneous final F-mask multiplication on mode 32"
                    if (kt == 2 and substituted_max <= 1.0e-20
                        and legacy_max >= 1.0e-10)
                    else (
                        "INCONCLUSIVE: prior trajectory inputs differ before kt=3"
                        if kt == 3 else "UNRESOLVED"
                    )
                ),
                "residual_disposition": (
                    "source-composition residue below 1e-20 m/s2"
                    if kt == 2 and substituted_max <= 1.0e-20 else
                    "the record lacks stage-3 input operands needed to separate "
                    "operator arithmetic from inherited trajectory debt"
                ),
            })
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
                boundary_candidates = {}
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
                    boundary_candidates[boundary] = boundary_fields
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

                # Compare the increment itself, not just the two endpoint
                # maxima.  Equal endpoint maxima do not prove that SBC added
                # no error because the unequal cells and signs may differ.
                sbc_increment_rows = []
                for field in ("T", "S"):
                    oracle_increment = (
                        tracer_oracle[f"after_sbc_{field}"][..., :30]
                        - tracer_oracle[f"after_advection_{field}"][..., :30]
                    )
                    candidate_increment = (
                        boundary_candidates["after_sbc"][field]
                        - boundary_candidates["after_advection"][field]
                    )
                    sbc_increment_rows.append(exact_score(
                        f"{case}.kt1.stage1.sbc_increment.{field}",
                        oracle_increment, candidate_increment, masks[field],
                    ))

                # One-variable experiment required by the scalar kt=2 stop
                # rule: inject NEMO's recorded stage-1 zF triplet into the
                # tracer consumer while leaving momentum and later stages on
                # production operands.  zFu/zFv are already exact, so this
                # changes only the observed zFw debt.
                oracle_transport = (
                    jnp.asarray(tracer_oracle["zFu"][..., :30]),
                    jnp.asarray(tracer_oracle["zFv"][..., :30]),
                    jnp.asarray(tracer_oracle["zFw"]),
                )
                override_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        stage1_tracer_transport_override=oracle_transport,
                    ),
                )
                override_state = override_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                override_state = host_and_clear(override_state)
                override_fields = lego_fields(override_state)
                oracle2 = read_entry(
                    oracle_root / "oracle_step_entry_kt00000002.bin",
                    dims=NEMO_DIMS,
                )
                override_rows = []
                for field in FIELDS:
                    reference = (
                        oracle2[field]
                        if field == "ssh" else oracle2[field][..., :30]
                    )
                    override_rows.append(exact_score(
                        f"{case}.kt2.NEMO_stage1_transport_override.{field}",
                        reference, override_fields[field], masks[field],
                    ))
                tracer_stage1_transport_discriminator = {
                    "changed_operand": (
                        "stage-1 tracer zFu/zFv/zFw triplet only; zFu/zFv "
                        "are independently exact, so the effective changed "
                        "operand is zFw"
                    ),
                    "sbc_increment_rows": sbc_increment_rows,
                    "NEMO_transport_override_rows": override_rows,
                }

                # The first prognostic split against scalar NEMO is stage-2
                # U/V, not the stage-1 tracer accumulator.  Compare NEMO's
                # source-ordered operator increments with legoESM's own
                # already-computed components, then substitute the complete
                # NEMO RHS at exactly the stage update boundary.
                term_oracle = read_stage2_terms(
                    oracle_root / "oracle_rkstage2_terms_kt00000001.bin",
                    dims=NEMO_DIMS,
                )

                def diagnostic_fields(**hook_args):
                    diagnostic_model = LatLonCGridOceanModel(
                        card.recipe.grid,
                        card.recipe.z_coord,
                        card.recipe.model_config,
                        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_args),
                    )
                    diagnostic = diagnostic_model.step(
                        state_before,
                        card.dt_s,
                        freshwater=freshwater,
                        surface_forcing=surface,
                    )
                    return lego_fields(host_and_clear(diagnostic))

                operator_rows = []
                boundaries = {
                    "hpg": ("before", "after_hpg"),
                    "vorticity": ("after_hpg", "after_vorticity"),
                    "advection": ("after_vorticity", "after_advection"),
                }
                for operator, (left, right) in boundaries.items():
                    exposed = diagnostic_fields(
                        expose_momentum_operator=operator,
                        expose_momentum_operator_stage=2,
                    )
                    rows = []
                    for field in ("u", "v"):
                        rows.append(accumulated_component_score(
                            f"{case}.kt1.stage2."
                            f"{operator}_accumulation.{field}",
                            term_oracle[f"{left}_{field}"][..., :30],
                            term_oracle[f"{right}_{field}"][..., :30],
                            exposed[field], masks[field],
                        ))
                    operator_rows.append({"operator": operator, "rows": rows})

                total_fields = diagnostic_fields(expose_stage2_momentum_rhs=True)
                total_rows = [
                    exact_score(
                        f"{case}.kt1.stage2.completed_rhs.{field}",
                        term_oracle[f"after_advection_{field}"][..., :30],
                        total_fields[field], masks[field],
                    )
                    for field in ("u", "v")
                ]
                oracle_rhs = _redundant_faces(
                    term_oracle["after_advection_u"],
                    term_oracle["after_advection_v"],
                )
                overridden_stage2 = diagnostic_fields(
                    stage2_momentum_rhs_override=(
                        jnp.asarray(oracle_rhs[0]), jnp.asarray(oracle_rhs[1])),
                    expose_momentum_stage=2,
                )
                stage2_oracle = read_stage(
                    oracle_root / "oracle_stage_kt00000001_s2.bin",
                    2,
                    dims=NEMO_DIMS,
                )
                override_stage_rows = [
                    exact_score(
                        f"{case}.kt1.stage2.NEMO_rhs_override.{field}",
                        stage2_oracle[field][..., :30],
                        overridden_stage2[field], masks[field],
                    )
                    for field in ("u", "v")
                ]
                composition_oracle = read_stage2_composition(
                    oracle_root / "oracle_rkstage2_operands_kt00000001.bin",
                    dims=NEMO_DIMS,
                )
                raw_fields = diagnostic_fields(expose_stage2_raw_momentum=True)
                overridden_raw = diagnostic_fields(
                    stage2_momentum_rhs_override=(
                        jnp.asarray(oracle_rhs[0]), jnp.asarray(oracle_rhs[1])),
                    expose_stage2_raw_momentum=True,
                )
                raw_rows = []
                override_raw_rows = []
                for field in ("u", "v"):
                    reference = composition_oracle[f"raw_Kaa_{field}"][..., :30]
                    raw_rows.append(exact_score(
                        f"{case}.kt1.stage2.raw_update.{field}",
                        reference, raw_fields[field], masks[field],
                    ))
                    override_raw_rows.append(exact_score(
                        f"{case}.kt1.stage2.NEMO_rhs_raw_update.{field}",
                        reference, overridden_raw[field], masks[field],
                    ))
                stage2_momentum_term_walk = {
                    "operator_increment_rows": operator_rows,
                    "completed_rhs_rows": total_rows,
                    "raw_update_rows": raw_rows,
                    "NEMO_completed_rhs_raw_update_rows": override_raw_rows,
                    "NEMO_completed_rhs_override_rows": override_stage_rows,
                }

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

            if deck == "r4":
                bt_oracle = read_bt_substeps(
                    oracle_root / "oracle_bt_substeps_kt00000001.bin",
                    expected_dims=NEMO_DIMS[:2],
                )
                trace_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_barotropic_substeps=True,
                    ),
                )
                trace = trace_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                trace = host_and_clear(trace)
                ordered = (
                    "eta_entry", "u_entry", "v_entry",
                    "eta_mid", "u_mid", "v_mid", "eta_exit", "eta_pgf",
                    "pgf_u", "pgf_v", "cor_u", "cor_v", "trd_u", "trd_v",
                    "slow_u", "slow_v", "u_exit", "v_exit",
                    "transport_metric_u", "transport_metric_v",
                )

                def bt_mask(name):
                    if name.startswith("eta"):
                        return masks["ssh"]
                    if name.startswith("u_") or name.endswith("_u"):
                        return masks["u"][..., 0]
                    return masks["v"][..., 0]

                first = None
                for substep in range(bt_oracle["ncycle"]):
                    for name in ordered:
                        reference = bt_oracle[name][substep]
                        candidate = _trace_native(bt_frame(trace.substeps, name), name)[substep]
                        row = exact_score(
                            f"{case}.kt1.bt.jn{substep + 1:02d}.{name}",
                            reference, candidate, bt_mask(name),
                        )
                        if row["status"] == "DEBT":
                            active = bt_mask(name)
                            unequal = reference[active].view(np.uint64) != candidate[active].view(np.uint64)
                            first_index = tuple(np.argwhere(active)[np.flatnonzero(unequal)[0]])
                            first = {
                                "substep": substep + 1,
                                "field": name,
                                "row": row,
                                "first_active_index": [int(index) for index in first_index],
                                "first_ulp": int(ulp_distance(
                                    np.asarray([reference[first_index]]),
                                    np.asarray([candidate[first_index]]),
                                )[0]),
                            }
                            break
                    if first is not None:
                        break

                # One-variable discriminator: replace only the frozen slow
                # forcing at the external-solver boundary with NEMO's recorded
                # value.  This is a private test hook, not a card selector.
                oracle_slow_u = np.asarray(trace.slow_forcing[1]).copy()
                oracle_slow_v = np.asarray(trace.slow_forcing[2]).copy()
                oracle_slow_u[:, 1:] = bt_oracle["slow_u"][0]
                oracle_slow_v[1:, :] = bt_oracle["slow_v"][0]
                control_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_barotropic_substeps=True,
                        barotropic_slow_forcing_override=(
                            jnp.asarray(oracle_slow_u), jnp.asarray(oracle_slow_v),
                        ),
                    ),
                )
                controlled = control_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                controlled = host_and_clear(controlled)
                control_rows = []
                for name in (
                    "slow_u", "slow_v", "u_exit", "v_exit",
                    "transport_metric_u", "transport_metric_v",
                ):
                    control_rows.append(exact_score(
                        f"{case}.kt1.bt.jn01.oracle_slow_override.{name}",
                        bt_oracle[name][0],
                        _trace_native(bt_frame(controlled.substeps, name), name)[0],
                        bt_mask(name),
                    ))
                depth_only_u = np.asarray(trace.slow_forcing_operands["depth_u"])[:, 1:]
                depth_only_v = np.asarray(trace.slow_forcing_operands["depth_v"])[1:, :]
                operand_u = trace.slow_forcing_operands
                wind_arms = {}
                for face, native, active in (
                    ("u", lambda value: np.asarray(value)[:, 1:], masks["u"][..., 0]),
                    ("v", lambda value: np.asarray(value)[1:, :], masks["v"][..., 0]),
                ):
                    arms = wind_expression_arms(
                        operand_u["wind_r1_rho0"],
                        native(operand_u[f"wind_tau_{face}"]),
                        native(operand_u[f"wind_r1_h{face}"]),
                        native(operand_u[f"H_{face}"]),
                        active,
                    )
                    wind_arms[face] = {
                        name: exact_score(
                            f"{case}.kt1.wind.{face}.{name}",
                            bt_oracle[f"slow_{face}"][0], value, active,
                        )
                        for name, value in arms.items()
                        if name != "stored_reciprocal"
                    }
                    production_slow = native(
                        trace.slow_forcing[1 if face == "u" else 2])
                    wind_arms[face]["production_slow"] = exact_score(
                        f"{case}.kt1.wind.{face}.production_slow",
                        bt_oracle[f"slow_{face}"][0], production_slow, active)
                    wind_arms[face]["reciprocal"] = exact_score(
                        f"{case}.kt1.wind.{face}.stored_vs_current_reciprocal",
                        arms["stored_reciprocal"],
                        native(operand_u[f"wind_r1_h{face}"]),
                        active,
                    )
                math_program = compiled_usrdef_math_program(
                    oracle_root / "nemo", expected_flavour=math_flavour)
                if math_flavour == "scalar":
                    wind_owner = (
                        "CLOSED_FOR_SCALAR_ORACLE: production slow and NEMO's "
                        "written left-associated expression are bit-exact; the "
                        "right-associated and reciprocal-first controls differ. "
                        "The admitted usrdef_sbc routine calls scalar libm sin."
                    )
                else:
                    wind_owner = (
                        "UPSTREAM_TAU_POLICY: production slow equals the "
                        "reconstructed left-associated scalar-libm composite, "
                        "but remains in debt to NEMO because the admitted "
                        "usrdef_sbc routine calls GNU vector sin"
                    )
                barotropic_resolution_walk = {
                    "first_difference": first,
                    "term": (
                        "stp2d wind term (r1_rho0 * utau) * r1_hu_0, already "
                        "inside frozen Ue_rhs/Ve_rhs before dynspg_ts substep 1"
                    ),
                    "raw_operand_ownership": wind_owner,
                    "rest_state_discriminator": {
                        "depth_only_u_max_abs": float(np.max(np.abs(depth_only_u))),
                        "depth_only_v_max_abs": float(np.max(np.abs(depth_only_v))),
                        "oracle_slow_override_rows": control_rows,
                    },
                    "wind_expression_discriminator": wind_arms,
                    "compiled_usrdef_math_program": math_program,
                    "arithmetic": {
                        "rn_Dt_s": card.dt_s,
                        "nn_e": card.recipe.model_config.barotropic.n_barotropic_substeps,
                        "substep_dt_s": card.dt_s / card.recipe.model_config.barotropic.n_barotropic_substeps,
                        "substep_reciprocal": 1.0 / (
                            card.dt_s / card.recipe.model_config.barotropic.n_barotropic_substeps
                        ),
                        "shipped_substep_dt_s": 14400.0 / 50.0,
                        "shipped_substep_reciprocal": 1.0 / (14400.0 / 50.0),
                        "grid_spacing_expression_m": 106000.0 / 4.0,
                        "grid_spacing_reciprocal": 1.0 / (106000.0 / 4.0),
                        "transport_weight_first_three": np.asarray(
                            trace.substeps["transport_weight"]
                        )[:3].tolist(),
                        "transport_weight_normalization": "r1_wgt2s from sum(wgtbtp2)",
                        "bottom_drag_at_substep1": "zero: entry velocity is rest",
                        "pgf_at_substep1": "bit-equal through pgf_u/pgf_v before slow_u/slow_v",
                    },
                }
    passed = all(row["status"] == "AT-BAR" for step in steps for row in step["rows"]) and all(
        control["status"] == "FIRED" for control in controls
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-exact-ladder-v1",
        "worktree": worktree_stamp(),
        "case": case,
        "deck": deck,
        "oracle_admission": admission,
        "oracle_math_flavour": math_flavour,
        "oracle_root": str(oracle_root),
        "ahm_oracle_root": str(ahm_root) if deck == "r4_smag" else None,
        "policy": "fp64/x64/cpu/libm certified",
        "acceptance": "zero unequal active float64 values in every field/step",
        "namelist": namelist._asdict(),
        "steps": steps,
        "rk_stage_walk": rk_stages,
        "tracer_stage1_transport": tracer_stage1_transport,
        "tracer_stage1_walk": tracer_stage1_walk,
        "tracer_stage1_transport_discriminator": (
            tracer_stage1_transport_discriminator),
        "stage2_momentum_term_walk": stage2_momentum_term_walk,
        "vertical_candidate": vertical_candidate,
        "barotropic_resolution_walk": barotropic_resolution_walk,
        "smagorinsky_coefficients": smagorinsky_coefficients,
        "smagorinsky_branch_census": smagorinsky_branch_census,
        "smagorinsky_operator": smagorinsky_operator,
        "controls": controls,
        "verdict": "PASS" if passed else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_stage2_composition_only(
    oracle_root: Path, output: Path, *, math_flavour: str = "scalar",
) -> dict:
    """Run the narrow source/update controls without the ten-step ladder."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree composition gate requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_oracle("r4", math_flavour, oracle_root)
    card = build_nemo_testcase_card(
        "GYRE-zco",
        namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=10, nn_stock=10, nn_write=10),
    )
    state = card.recipe.initial_state
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    terms = read_stage2_terms(
        oracle_root / "oracle_rkstage2_terms_kt00000001.bin",
        dims=NEMO_DIMS,
    )
    composition = read_stage2_composition(
        oracle_root / "oracle_rkstage2_operands_kt00000001.bin",
        dims=NEMO_DIMS,
    )
    oracle_rhs = _redundant_faces(
        terms["after_advection_u"], terms["after_advection_v"])

    def run_model(**hook_args):
        model = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_args),
        )
        diagnostic = model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        return jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf)
            if isinstance(leaf, jax.Array) else leaf,
            diagnostic,
        )

    masks = expected_masks(card)
    production_raw = lego_fields(run_model(expose_stage2_raw_momentum=True))
    overridden_raw = lego_fields(run_model(
        stage2_momentum_rhs_override=(
            jnp.asarray(oracle_rhs[0]), jnp.asarray(oracle_rhs[1])),
        expose_stage2_raw_momentum=True,
    ))
    production_rows = []
    override_rows = []
    controls = []
    for field in ("u", "v"):
        reference = composition[f"raw_Kaa_{field}"][..., :30]
        production_rows.append(exact_score(
            f"GYRE-zco-r4.kt1.stage2.production_raw_update.{field}",
            reference, production_raw[field], masks[field],
        ))
        override_rows.append(exact_score(
            f"GYRE-zco-r4.kt1.stage2.NEMO_rhs_raw_update.{field}",
            reference, overridden_raw[field], masks[field],
        ))
        planted = exact_score(
            f"control.one_ulp.raw_update.{field}",
            reference, reference, masks[field], plant_one_ulp=True,
        )
        controls.append({
            **planted,
            "status": "FIRED" if planted["n_unequal"] == 1 else "MISSED",
        })

    # Prove the private RHS injection is consumed, rather than merely
    # returning a coincidentally exact diagnostic. Perturb one wet native-U
    # RHS value by a material fixed amount and require the raw update to move.
    planted_rhs = [np.array(value, copy=True) for value in oracle_rhs]
    wet_u_index = tuple(int(value) for value in np.argwhere(masks["u"])[0])
    internal_u_index = (
        wet_u_index[0], wet_u_index[1] + 1, wet_u_index[2])
    rhs_delta = np.float64(1.0e-6)
    planted_rhs[0][internal_u_index] += rhs_delta
    planted_raw = lego_fields(run_model(
        stage2_momentum_rhs_override=(
            jnp.asarray(planted_rhs[0]), jnp.asarray(planted_rhs[1])),
        expose_stage2_raw_momentum=True,
    ))
    rhs_effect_rows = [
        exact_score(
            f"control.material_rhs_override.{field}",
            overridden_raw[field], planted_raw[field], masks[field],
        )
        for field in ("u", "v")
    ]
    rhs_effect_fired = any(row["n_unequal"] > 0 for row in rhs_effect_rows)

    # Score each isolated legoESM term through the actual accumulator
    # boundary. This avoids attempting to reconstruct NEMO's addend by
    # subtracting two rounded accumulator snapshots.
    boundaries = {
        "hpg": ("before", "after_hpg"),
        "vorticity": ("after_hpg", "after_vorticity"),
        "advection": ("after_vorticity", "after_advection"),
    }
    accumulation_rows = []
    for operator, (left, right) in boundaries.items():
        exposed = lego_fields(run_model(
            expose_momentum_operator=operator,
            expose_momentum_operator_stage=2,
        ))
        rows = [
            accumulated_component_score(
                f"GYRE-zco-r4.kt1.stage2."
                f"{operator}_accumulation.{field}",
                terms[f"{left}_{field}"][..., :30],
                terms[f"{right}_{field}"][..., :30],
                exposed[field], masks[field],
            )
            for field in ("u", "v")
        ]
        accumulation_rows.append({"boundary": operator, "rows": rows})
    first_accumulation_debt = next(
        (
            {
                "boundary": group["boundary"],
                "fields": [
                    row["name"].rsplit(".", 1)[-1]
                    for row in group["rows"] if row["status"] == "DEBT"
                ],
            }
            for group in accumulation_rows
            if any(row["status"] == "DEBT" for row in group["rows"])
        ),
        None,
    )

    # Prove the stage-1 zF substitution reaches the tracer consumer. The
    # scientific arm substitutes NEMO's zFw and does not move the kt=2
    # maxima; this control changes one active zFw by 1 m3/s and requires a
    # downstream prognostic tracer to move.
    tracer_stage1 = read_tracer_stage12(
        oracle_root / "oracle_rktracer_operands_kt00000001_s1.bin",
        1, dims=NEMO_DIMS,
    )
    transport = [
        np.array(tracer_stage1[name][..., :30]
                 if name != "zFw" else tracer_stage1[name], copy=True)
        for name in ("zFu", "zFv", "zFw")
    ]
    baseline_transport_state = lego_fields(run_model(
        stage1_tracer_transport_override=tuple(jnp.asarray(x) for x in transport),
    ))
    zfw_index = next(
        tuple(int(value) for value in index)
        for index in np.argwhere(masks["T"])
        if index[2] > 0
    )
    transport[2][zfw_index] += np.float64(1.0)
    planted_transport_state = lego_fields(run_model(
        stage1_tracer_transport_override=tuple(jnp.asarray(x) for x in transport),
    ))
    zfw_effect_rows = [
        exact_score(
            f"control.material_zFw_override.{field}",
            baseline_transport_state[field], planted_transport_state[field],
            masks[field],
        )
        for field in ("T", "S")
    ]
    zfw_effect_fired = any(row["n_unequal"] > 0 for row in zfw_effect_rows)

    controls.extend([
        {
            "name": "control.material_rhs_override",
            "status": "FIRED" if rhs_effect_fired else "MISSED",
            "delta": float(rhs_delta),
            "index": list(internal_u_index),
            "rows": rhs_effect_rows,
        },
        {
            "name": "control.material_zFw_override",
            "status": "FIRED" if zfw_effect_fired else "MISSED",
            "delta_m3_s": 1.0,
            "index": list(zfw_index),
            "rows": zfw_effect_rows,
        },
    ])
    report = {
        "format": "nemo-testcase-l2-gyre-r4-narrow-discriminator-v2",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "policy": "fp64/x64/cpu/libm certified",
        "changed_operand": "complete NEMO stage-2 momentum RHS only",
        "comparison_boundary": (
            "raw Kaa immediately after the RK update and before barotropic "
            "mean replacement"
        ),
        "production_raw_rows": production_rows,
        "NEMO_rhs_override_raw_rows": override_rows,
        "source_order_accumulation_rows": accumulation_rows,
        "first_nonexact_accumulation_boundary": first_accumulation_debt,
        "mean_replacement_boundary": {
            "status": "OPEN_INPUT_VERSUS_ARITHMETIC",
            "reason": (
                "the NEMO-RHS raw update is exact, while the recorded post-"
                "replacement state is not; recorded barotropic targets and "
                "zub/zvb have not yet been substituted"
            ),
        },
        "controls": controls,
        "verdict": (
            "RAW_UPDATE_AT_BAR"
            if all(row["status"] == "AT-BAR" for row in override_rows)
            and all(control["status"] == "FIRED" for control in controls)
            else "DEBT"
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_stage3_geometry_only(
    oracle_root: Path, output: Path, *, math_flavour: str = "scalar",
) -> dict:
    """Bound a post-hoc ratio replay; do not call it executed NEMO arithmetic."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree geometry gate requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_oracle("r4", math_flavour, oracle_root)
    card = build_nemo_testcase_card(
        "GYRE-zco",
        namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=10, nn_stock=10, nn_write=10),
    )
    state = card.recipe.initial_state
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(tracer_process_trace=()),
    )
    trace = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    trace = jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf)
        if isinstance(leaf, jax.Array) else leaf,
        trace,
    )
    oracle = read_tracer_stage3_hires(
        oracle_root / "oracle_rktracer_stage3_kt00000001.bin",
        dims=NEMO_DIMS,
    )
    masks = expected_masks(card)
    mask3 = masks["T"]
    mask2 = np.any(mask3, axis=-1)
    tbb = np.asarray(trace.Tbb)[..., :30]
    qbb = np.asarray(trace.q_Kbb)
    qmm = np.asarray(trace.q_Kmm)
    qaa = np.asarray(trace.q_Kaa)
    oracle_tbb = oracle["Kbb_T"][..., :30]
    oracle_qbb = np.float64(1.0) + oracle["r3t_Kbb"]
    oracle_qmm = np.float64(1.0) + oracle["r3t_Kmm"]
    oracle_qaa = np.float64(1.0) + oracle["r3t_Kaa"]
    operand_rows = [
        exact_score("stage3.geometry.Tbb", oracle_tbb, tbb, mask3),
        exact_score("stage3.geometry.q_Kbb", oracle_qbb, qbb, mask2),
        exact_score("stage3.geometry.q_Kmm", oracle_qmm, qmm, mask2),
        exact_score("stage3.geometry.q_Kaa", oracle_qaa, qaa, mask2),
    ]
    oracle_geometry = (
        oracle_qbb[..., None] * oracle_tbb / oracle_qaa[..., None])
    literal_ratio = qbb[..., None] * tbb / qaa[..., None]
    e3t_0 = np.asarray(card.recipe.z_coord.nemo_e3t_0)[..., :30]
    thickness_association = (
        (e3t_0 * qbb[..., None]) * tbb
        / (e3t_0 * qaa[..., None]))
    production_geometry = np.asarray(trace.boundaries[0])[..., :30]
    composition_rows = [
        exact_score(
            "stage3.geometry.literal_ratio_replay",
            oracle_geometry, literal_ratio, mask3),
        exact_score(
            "stage3.geometry.production_reference_thickness",
            oracle_geometry, production_geometry, mask3),
        exact_score(
            "stage3.geometry.reference_thickness_reconstruction",
            production_geometry, thickness_association, mask3),
    ]
    planted_qbb = np.array(qbb, copy=True)
    plant_index = tuple(int(value) for value in np.argwhere(mask2)[0])
    planted_qbb[plant_index] = np.nextafter(
        planted_qbb[plant_index], np.inf)
    planted_literal = planted_qbb[..., None] * tbb / qaa[..., None]
    plant = exact_score(
        "control.one_ulp.q_Kbb_replay",
        literal_ratio, planted_literal, mask3)
    replay_bounded = (
        all(row["status"] == "AT-BAR" for row in operand_rows)
        and composition_rows[0]["status"] == "AT-BAR"
        and composition_rows[2]["status"] == "AT-BAR"
        and plant["status"] == "DEBT")
    report = {
        "format": "nemo-testcase-l2-gyre-r4-stage3-geometry-v1",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "policy": "fp64/x64/cpu/libm certified",
        "operand_rows": operand_rows,
        "composition_rows": composition_rows,
        "one_variable_replay": {
            "changed_expression": (
                "remove the algebraically cancelling e3t_0 factor from the "
                "stage-3 Kbb-to-Kaa geometry association"
            ),
            "control": {
                **plant,
                "status": "FIRED" if plant["status"] == "DEBT" else "MISSED",
                "index": list(plant_index),
            },
        },
        "interpretation": (
            "POSTHOC_RATIO_REPLAY_ONLY: compiled trazdf.f90:284-289 "
            "retains e3t_0 in the executed RHS; the current record does not "
            "contain its matrix/RHS operands"
        ),
        "owner": "UNMEASURED_EXECUTED_TRAZDF_OPERANDS",
        "verdict": "REPORT_ONLY" if replay_bounded else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=tuple(DEFAULT_ORACLES), default="r4")
    parser.add_argument(
        "--math-flavour", choices=ORACLE_FLAVOURS, default="vector")
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--ahm-root", type=Path, default=DEFAULT_AHM_ROOT)
    parser.add_argument("--stage2-composition-only", action="store_true")
    parser.add_argument("--stage3-geometry-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        admission = admit_oracle(args.deck, args.math_flavour, args.oracle_root)
        if args.stage2_composition_only and args.stage3_geometry_only:
            raise ValueError("choose only one narrow discriminator")
        if args.stage2_composition_only:
            if args.deck != "r4":
                raise ValueError("stage-2 composition-only gate requires deck r4")
            report = run_stage2_composition_only(
                Path(admission["root"]), args.output,
                math_flavour=args.math_flavour,
            )
        elif args.stage3_geometry_only:
            if args.deck != "r4":
                raise ValueError("stage-3 geometry discriminator requires deck r4")
            report = run_stage3_geometry_only(
                Path(admission["root"]), args.output,
                math_flavour=args.math_flavour,
            )
        else:
            report = run(
                Path(admission["root"]), args.output,
                deck=args.deck, ahm_root=args.ahm_root,
                math_flavour=args.math_flavour,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["verdict"] in (
        "PASS", "RAW_UPDATE_AT_BAR", "REPORT_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
