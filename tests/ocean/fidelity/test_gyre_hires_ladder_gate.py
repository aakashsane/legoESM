import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_ladder_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_hires_ladder_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_ladder_decks_have_distinct_preregistered_oracle_roots():
    assert set(GATE.DEFAULT_ORACLES) == {"r4", "r4_smag"}
    assert "round1/oracle_gyre_r4_stage" in str(GATE.DEFAULT_ORACLES["r4"])
    assert "round2/oracle_gyre_r4_smag_stage" in str(
        GATE.DEFAULT_ORACLES["r4_smag"])


def test_all_math_flavour_oracles_are_admitted_by_content():
    for deck in ("r4", "r4_smag"):
        for flavour in ("scalar", "vector"):
            admitted = GATE.admit_oracle(deck, flavour)
            assert admitted["deck"] == deck
            assert admitted["math_flavour"] == flavour
            assert len(admitted["checked_files"]) == 12


def test_oracle_admission_refuses_root_and_digest_mutations(tmp_path):
    with pytest.raises(ValueError, match="unregistered oracle root"):
        GATE.admit_oracle("r4", "scalar", tmp_path)

    registry = json.loads(GATE.ADMISSIONS_PATH.read_text(encoding="utf-8"))
    registry["admissions"]["r4:scalar"]["files"]["nemo"] = "0" * 64
    planted = tmp_path / "admissions.json"
    planted.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch for nemo"):
        GATE.admit_oracle("r4", "scalar", admissions_path=planted)


def test_admitted_usrdef_binary_exposes_vector_and_scalar_sin_cos_no_exp():
    executable = GATE.DEFAULT_ORACLES["r4"] / "nemo"
    if not executable.is_file():
        pytest.skip("admitted r4 NEMO executable is unavailable")
    program = GATE.compiled_usrdef_math_program(executable)
    assert program["sha256"] == (
        "974dc326b4f5237a0e8dfe50ba0ae8a4573507d57429fb82069ff3c20abfad41")
    assert any(name.endswith("sin") for name in program["vector_sin_calls"])
    assert program["scalar_sin_calls"] == ["sin"]
    assert {name.rsplit("_", 1)[-1]
            for name in program["vector_transcendental_calls"]} == {"sin", "cos"}
    assert set(program["scalar_transcendental_calls"]) == {"sin", "cos"}


def test_admitted_scalar_usrdef_binary_refuses_vector_math():
    admitted = GATE.admit_oracle("r4", "scalar")
    program = GATE.compiled_usrdef_math_program(
        Path(admitted["root"]) / "nemo", expected_flavour="scalar")
    assert program["math_flavour"] == "scalar"
    assert program["vector_sin_calls"] == []
    assert program["scalar_sin_calls"] == ["sin"]
    assert program["vector_transcendental_calls"] == []
    assert set(program["scalar_transcendental_calls"]) == {"sin", "cos"}
    with pytest.raises(ValueError, match="vector oracle"):
        GATE.compiled_usrdef_math_program(
            Path(admitted["root"]) / "nemo", expected_flavour="vector")


def test_usrdef_math_inventory_refuses_missing_binary(tmp_path):
    with pytest.raises(FileNotFoundError, match="admitted NEMO executable"):
        GATE.compiled_usrdef_math_program(tmp_path / "nemo")


def test_exact_score_one_ulp_control_fires():
    values = np.array([[0.0, 1.0], [-2.0, 3.0]], dtype=np.float64)
    mask = np.ones_like(values, dtype=bool)
    assert GATE.exact_score("base", values, values, mask)["status"] == "AT-BAR"
    planted = GATE.exact_score(
        "plant", values, values, mask, plant_one_ulp=True)
    assert planted["status"] == "DEBT"
    assert planted["n_unequal"] == 1
    assert planted["max_ulp"] == 1
    assert planted["ulp_at_max_abs"] == 1
    assert planted["first_unequal_index"] == [0, 0]
    assert planted["max_abs_index"] == [0, 0]


def test_scalar_first_owner_discriminator_scores_increment_and_narrow_override():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'stage1_tracer_transport_override=oracle_transport' in source
    assert 'tracer_oracle["zFu"][..., :30]' in source
    assert 'tracer_oracle["zFv"][..., :30]' in source
    assert 'tracer_oracle["zFw"]' in source
    assert 'after_sbc_{field}' in source and 'after_advection_{field}' in source
    assert 'NEMO_stage1_transport_override' in source


def test_wind_expression_arms_preserve_compiled_association_and_reciprocal():
    rho = np.float64(1.0 / 1026.0)
    stress = np.array([[-0.2, 0.0]], dtype=np.float64)
    depth = np.array([[4194.0, 1.0]], dtype=np.float64)
    mask = np.array([[1.0, 0.0]], dtype=np.float64)
    current = np.array([[np.float64(1.0) / depth[0, 0], 0.0]])
    arms = GATE.wind_expression_arms(rho, stress, current, depth, mask)
    expected = (rho * stress[0, 0]) * (
        mask[0, 0] / ((depth[0, 0] + 1.0) - mask[0, 0]))
    assert arms["nemo_written"][0, 0].view(np.uint64) == expected.view(np.uint64)
    assert (
        arms["right_associated"][0, 0].view(np.uint64)
        != arms["nemo_written"][0, 0].view(np.uint64)
    )
    assert np.array_equal(arms["stored_reciprocal"][:, 1], np.zeros(1))


def test_smagorinsky_branch_census_is_exhaustive_and_upper_owns_final_min():
    diagnostics = {
        "t_lower_clamped": np.array([True, False, True, False]),
        "t_upper_clamped": np.array([False, False, True, True]),
        "f_lower_clamped": np.array([True, False, False]),
        "f_upper_clamped": np.array([False, False, True]),
    }
    census = GATE.coefficient_branch_census(
        diagnostics, np.ones(4, dtype=bool), np.ones(3, dtype=bool))
    assert census["T"]["counts"] == {
        "lower_limit": 1, "deformation": 1, "upper_limit": 2}
    assert census["F"]["counts"] == {
        "lower_limit": 1, "deformation": 1, "upper_limit": 1}
    assert sum(census["T"]["fractions"].values()) == 1.0


@pytest.mark.timeout(900)
def test_r4_wind_grouping_does_not_explain_oracle_slow_debt():
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_card_mesh_operands,
        nemo_qco_live_face_geometry_from_operands,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(
        "GYRE-zco", namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=1, nn_stock=1, nn_write=1))
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    trace = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)

    dtype = state.u.data.dtype
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(dtype)
    umask3, vmask3 = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    ops = nemo_qco_card_mesh_operands(
        h_ref, umask3.astype(dtype), vmask3.astype(dtype),
        card.recipe.grid, dtype)
    live = nemo_qco_live_face_geometry_from_operands(
        state.eta.data, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3,
        ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v)

    got_u = np.asarray(trace.slow_forcing_operands["wind_r1_hu"])[:, 1:]
    got_v = np.asarray(trace.slow_forcing_operands["wind_r1_hv"])[1:, :]
    expected_u = np.asarray(live.r1_hu)
    expected_v = np.asarray(live.r1_hv)
    wet_u = np.asarray(state.u_mask.data[:, 1:], dtype=bool)
    wet_v = np.asarray(state.v_mask.data[1:, :], dtype=bool)
    assert np.array_equal(
        got_u[wet_u].view(np.uint64), expected_u[wet_u].view(np.uint64))
    assert np.array_equal(
        got_v[wet_v].view(np.uint64), expected_v[wet_v].view(np.uint64))

    # At the rest state the algebraic reciprocal happens to round identically;
    # this is measured explicitly rather than used as a non-vacuous plant.
    old_u = np.asarray(1.0 / trace.slow_forcing_operands["H_u"])[:, 1:]
    assert np.array_equal(
        old_u[wet_u].view(np.uint64), expected_u[wet_u].view(np.uint64))

    oracle = GATE.read_bt_substeps(
        GATE.DEFAULT_ORACLES["r4"] / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=GATE.NEMO_DIMS[:2])
    actual_slow = np.asarray(trace.slow_forcing[1])[:, 1:]
    arms = GATE.wind_expression_arms(
        trace.slow_forcing_operands["wind_r1_rho0"],
        np.asarray(trace.slow_forcing_operands["wind_tau_u"])[:, 1:],
        got_u,
        np.asarray(trace.slow_forcing_operands["H_u"])[:, 1:],
        wet_u,
    )
    scores = {
        name: GATE.exact_score(name, oracle["slow_u"][0], value, wet_u)
        for name, value in {
            "production_slow": actual_slow,
            "left_associated_wind": arms["nemo_written"],
            "right_associated_wind": arms["right_associated"],
            "reciprocal_first_wind": arms["reciprocal_first"],
        }.items()
    }
    assert np.array_equal(
        actual_slow[wet_u].view(np.uint64),
        arms["nemo_written"][wet_u].view(np.uint64))
    assert all(row["status"] == "DEBT" for row in scores.values())
    assert scores["production_slow"]["n_unequal"] == 4158
    assert scores["production_slow"]["ulp_at_max_abs"] == 2


def test_shared_entry_reader_accepts_runtime_high_resolution_dimensions(tmp_path):
    dims = (6, 6, 2)
    count = np.prod(dims)
    payload = np.arange(4 * count + dims[0] * dims[1], dtype=np.float64)
    path = tmp_path / "oracle_step_entry_kt00000001.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L1_ENTRY_1 ")
        stream.write(struct.pack("=8i", 1, 1, 1, *dims, 2, 64))
        payload.tofile(stream)
    record = GATE.read_entry(path, dims=dims)
    assert record["kt"] == 1 and record["Nbb"] == 1
    assert record["T"].shape == (2, 2, 2)
    assert record["ssh"].shape == (2, 2)


def test_shared_stage_and_zdf_readers_accept_runtime_dimensions(tmp_path):
    dims = (6, 6, 2)
    count = np.prod(dims)
    stage_path = tmp_path / "oracle_stage_kt00000001_s1.bin"
    with stage_path.open("wb") as stream:
        stream.write(b"NEMO_L1_STAGE_1 ")
        stream.write(struct.pack("=9i", 1, 1, 1, 3, *dims, 2, 64))
        np.arange(4 * count + dims[0] * dims[1], dtype=np.float64).tofile(stream)
    stage = GATE.read_stage(stage_path, 1, dims=dims)
    assert stage["T"].shape == (2, 2, 2)

    zdf_path = tmp_path / "oracle_zdf_entry_kt00000001.bin"
    with zdf_path.open("wb") as stream:
        stream.write(b"NEMO_L2_ZDF___1 ")
        stream.write(struct.pack("=7i", 1, 1, 1, *dims, 64))
        np.arange(count + 2 * 2 * 2, dtype=np.float64).tofile(stream)
    zdf = GATE.read_zdf_entry(zdf_path, dims=dims)
    assert zdf["avm"].shape == (2, 2, 2)
    assert zdf["avt"].shape == (2, 2, 2)


def test_shared_tracer_stage_reader_accepts_runtime_dimensions(tmp_path):
    dims = (6, 6, 2)
    count = np.prod(dims)
    path = tmp_path / "oracle_rktracer_operands_kt00000001_s1.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L2_RKTRA_1 ")
        stream.write(struct.pack("=11i", 1, 1, 1, 1, 1, 3, 3, *dims, 64))
        np.arange(15 * count + 3 * dims[0] * dims[1], dtype=np.float64).tofile(stream)
    record = GATE.read_tracer_stage12(path, 1, dims=dims)
    assert record["after_advection_T"].shape == (2, 2, 2)
    assert record["after_sbc_S"].shape == (2, 2, 2)


def test_ahm_and_dynldf_readers_validate_headers_and_staggering(tmp_path):
    dims = (6, 6, 3)
    count = int(np.prod(dims))
    ahm_path = tmp_path / "oracle_ahm_kt00000002.bin"
    with ahm_path.open("wb") as stream:
        stream.write(b"NEMO_L4_AHMTF__1")
        stream.write(struct.pack("=7i", 1, 2, 3, *dims, 64))
        np.arange(2 * count, dtype=np.float64).tofile(stream)
    ahm = GATE.read_ahm(ahm_path, expected_kt=2, dims=dims)
    assert ahm["ahmt"].shape == (2, 2, 2)
    assert ahm["ahmf"].shape == (3, 3, 2)
    assert np.all(ahm["ahmf"][0] == 0.0)

    dyn_path = tmp_path / "oracle_dynldf_kt00000002.bin"
    with dyn_path.open("wb") as stream:
        stream.write(b"NEMO_L4_DYNLDF_1")
        stream.write(struct.pack("=10i", 1, 2, 3, 3, 2, 1, *dims, 64))
        np.arange(4 * count, dtype=np.float64).tofile(stream)
    dyn = GATE.read_dynldf(dyn_path, expected_kt=2, dims=dims)
    assert dyn["u_before"].shape == (2, 2, 2)
    assert dyn["Kbb"] == 3 and dyn["Kmm"] == 2 and dyn["Krhs"] == 1

    bad = tmp_path / "bad.bin"
    bad.write_bytes(ahm_path.read_bytes().replace(b"NEMO_L4_AHMTF__1", b"NEMO_L4_AHMTF__X", 1))
    with np.testing.assert_raises_regex(ValueError, "magic"):
        GATE.read_ahm(bad, expected_kt=2, dims=dims)
