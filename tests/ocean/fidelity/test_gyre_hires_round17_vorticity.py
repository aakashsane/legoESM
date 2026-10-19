"""Round-17 common-state ENE-q discriminator controls."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/validate/ocean_fidelity/testcases"
TOOL = SCRIPTS / "nemo_testcase_l2_gyre_hires_round17_vorticity.py"
RUN = (SCRIPTS / "nemo_testcase_l2_gyre_hires_round17_vorticity" / "run.sh")


def _load_tool():
    name = "_test_gyre_hires_round17_vorticity"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stage_and_q_readers_refuse_layout_and_preserve_fortran_order(
    tmp_path, monkeypatch,
):
    tool = _load_tool()
    monkeypatch.setattr(tool, "NATIVE_DIMS", (8, 7, 3))
    monkeypatch.setattr(tool, "ACTIVE_LEVELS", 2)
    nx, ny, nz = tool.NATIVE_DIMS
    n3 = nx * ny * nz
    arrays = [np.arange(n3, dtype=np.float64) + 1000 * i for i in range(4)]
    stage = tmp_path / tool.STAGE_NAME
    stage.write_bytes(
        b"NEMO_L2_R17VOR1 "
        + struct.pack("=11i", 1, 4321, 3, 1, 2, 3, 3, nx, ny, nz, 64)
        + b"".join(array.tobytes() for array in arrays)
    )
    got = tool.read_stage_vorticity(stage)
    expected = arrays[1].reshape((nx, ny, nz), order="F").transpose(1, 0, 2)
    np.testing.assert_array_equal(got["before_v"], expected)

    qpath = tmp_path / "q.bin"
    slabs = []
    for level in range(tool.ACTIVE_LEVELS):
        for field in range(3):
            slabs.append(np.arange(nx * ny, dtype=np.float64)
                         + 100 * level + 10 * field)
    qpath.write_bytes(
        b"NEMO_L2_R17Q__1 "
        + struct.pack("=8i", 1, 4321, 2, 4, nx, ny,
                      tool.ACTIVE_LEVELS, 64)
        + b"".join(array.tobytes() for array in slabs)
    )
    operands = tool.read_ene_operands(qpath)
    expected_q0 = slabs[0].reshape((nx, ny), order="F").T
    np.testing.assert_array_equal(operands["q"][..., 0], expected_q0)

    planted = bytearray(qpath.read_bytes())
    planted[:16] = b"WRONG_MAGIC____1"
    qpath.write_bytes(planted)
    with pytest.raises(Exception, match="bad magic"):
        tool.read_ene_operands(qpath)


def test_q_reader_ignores_unread_halo_but_refuses_consumed_nonfinite(
    tmp_path, monkeypatch,
):
    """The guard follows dynvor's loop bounds, not uninitialised work memory."""
    tool = _load_tool()
    monkeypatch.setattr(tool, "NATIVE_DIMS", (8, 7, 3))
    monkeypatch.setattr(tool, "ACTIVE_LEVELS", 2)
    nx, ny, _ = tool.NATIVE_DIMS
    slabs = [np.ones(nx * ny, dtype=np.float64) for _ in range(6)]
    slabs[0][0] = np.nan  # q(1,1): outside every executed/consumed loop bound.

    def write(path):
        path.write_bytes(
            b"NEMO_L2_R17Q__1 "
            + struct.pack("=8i", 1, 4321, 2, 4, nx, ny,
                          tool.ACTIVE_LEVELS, 64)
            + b"".join(array.tobytes() for array in slabs)
        )

    qpath = tmp_path / "q.bin"
    write(qpath)
    tool.read_ene_operands(qpath)
    slabs[0][nx + 1] = np.nan  # q(j=2,i=2): consumed by the live stencil.
    write(qpath)
    with pytest.raises(Exception, match="non-finite consumed q word"):
        tool.read_ene_operands(qpath)


def test_common_card_keeps_registered_deck_run_controls():
    """The developed-step number is forcing time, not a card namelist edit."""
    tool = _load_tool()
    card = tool._common_card()
    namelist = card.resolved_namelist
    assert namelist.nn_it000 == 1
    assert namelist.nn_itend == 26280
    assert namelist.nn_GYRE == 4
    assert namelist.rn_Dt_s == 3600.0
    assert namelist.nn_ahm_ijk_t == 32


def test_substitution_verdict_requires_twin_envelope_and_untouched_rows():
    tool = _load_tool()
    selection = np.ones((2, 3, 1), dtype=bool)
    vector = np.zeros(selection.shape)
    scalar = np.full(selection.shape, 2.0)
    baseline = np.full(selection.shape, 20.0)
    substituted = np.full(selection.shape, 2.5)
    report = tool.substitution_verdict(
        baseline, substituted, vector, scalar, selection, True)
    assert report["verdict"] == "CONFIRM_Q_FORMATION"
    assert report["baseline_candidate_to_twin_ratio"] == 10.0

    outside = tool.substitution_verdict(
        baseline, np.full(selection.shape, 4.01), vector, scalar,
        selection, True)
    assert outside["verdict"] == "REFUTE_Q_FORMATION"
    moved = tool.substitution_verdict(
        baseline, substituted, vector, scalar, selection, False)
    assert moved["verdict"] == "REFUTE_Q_FORMATION"


def test_model_tendency_q_override_moves_only_vorticity_component():
    """The shared path consumes the oracle q without rebuilding its stencil."""
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("GYRE-zco")
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    jj_u, ii_u, kk_u = np.indices(state.u.data.shape)
    jj_v, ii_v, kk_v = np.indices(state.v.data.shape)
    u = (1.0e-3 * (1 + jj_u + 2 * ii_u + kk_u)
         * np.asarray(state.u_mask.data)[..., None])
    v = (-7.0e-4 * (2 + 2 * jj_v + ii_v + kk_v)
         * np.asarray(state.v_mask.data)[..., None])
    u[:, -1, :] = u[:, 0, :]
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
    )
    baseline = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        return_nemo_operator_components=True)
    q_shape = (state.T.data.shape[0] + 1,
               state.T.data.shape[1] + 1, state.T.data.shape[2])
    overridden = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        nemo_vorticity_q_override=jnp.zeros(q_shape, dtype=jnp.float64),
        return_nemo_operator_components=True)
    baseline_components = baseline[2]
    override_components = overridden[2]
    assert not np.array_equal(
        np.asarray(baseline_components["vorticity_v"].data),
        np.asarray(override_components["vorticity_v"].data),
    )
    for name in ("hpg_v", "advection_v", "operand_velocity_v",
                 "operand_momentum_h_v"):
        left = getattr(baseline_components[name], "data",
                       baseline_components[name])
        right = getattr(override_components[name], "data",
                        override_components[name])
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_operator_script_pins_common_restart_and_same_binary_control():
    source = RUN.read_text(encoding="utf-8")
    for needle in (
        "--run-vector", "--run-scalar", "-d 'OCE'", "RANKS must be 1",
        "COMMON_RESTART_SHA256=a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12",
        "run_nemo control 0", "run_nemo instrumented 1",
        "REFUSE_WRITER_MOVED_TRAJECTORY", "GYRE_R17_WRITE",
        "PREFLIGHT_OK",
    ):
        assert needle in source
    assert "rg " not in source
    model_source = (ROOT / "packages/ocean/legoesm/ocean/dynamics/"
                    "ocean_model_latlon_cgrid.py").read_text(encoding="utf-8")
    assert "if stage_index == 3 else None" in model_source
