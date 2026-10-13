import importlib.util
import struct
from pathlib import Path

import numpy as np

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
