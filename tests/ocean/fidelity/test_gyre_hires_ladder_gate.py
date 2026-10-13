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
