from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts/validate/ocean_fidelity/testcases"
SCRIPT = SCRIPT_DIR / "nemo_testcase_l2_gyre_hires_round20_stage1.py"


def _load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("gyre_round20_stage1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_envelope_score_and_one_ulp_plant_fire():
    module = _load_module()
    scalar = np.ones((2, 2), dtype=np.float64)
    vector = scalar.copy()
    candidate = scalar.copy()
    mask = np.ones_like(scalar, dtype=bool)

    exact = module.envelope_score(candidate, scalar, vector, mask)
    assert exact["legoesm_minus_scalar"]["bit_exact"]
    assert not exact["clears_four_times_twin"]

    planted = module._plant_one_ulp(candidate, mask)
    moved = module.envelope_score(planted, scalar, vector, mask)
    assert moved["legoesm_minus_scalar"]["unequal_words"] == 1
    assert moved["clears_four_times_twin"]


def test_envelope_uses_registered_four_times_maximum():
    module = _load_module()
    scalar = np.zeros(2, dtype=np.float64)
    vector = np.array([1.0, 0.0], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    below = module.envelope_score(
        np.array([3.0, 0.0]), scalar, vector, mask)
    at_bar = module.envelope_score(
        np.array([4.0, 0.0]), scalar, vector, mask)
    assert not below["clears_four_times_twin"]
    assert at_bar["clears_four_times_twin"]
    assert at_bar["candidate_to_twin_max_ratio"] == 4.0


def test_first_clearing_stage_respects_stage_and_face_order():
    module = _load_module()

    def row(clears):
        return {"clears_four_times_twin": clears}

    stages = [
        {"stage": 1, "rows": {"u": row(False), "v": row(True)}},
        {"stage": 2, "rows": {"u": row(True), "v": row(True)}},
    ]
    result = module.first_clearing_stage(stages)
    assert result is not None
    assert (result["stage"], result["field"]) == (1, "v")


def test_active_level_mapping_drops_only_nemo_closed_jpk(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (6, 5, 4))
    monkeypatch.setattr(module, "ACTIVE_LEVELS", 3)
    field = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    active = module._active_levels(field)
    assert active.shape == (2, 3, 3)
    np.testing.assert_array_equal(active, field[..., :3])
    with pytest.raises(Exception, match="unexpected shape"):
        module._active_levels(field[..., :3])


def test_completed_rhs_reader_rejects_bad_header(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (3, 2, 2))
    count = 3 * 2 * 2
    fields = np.arange(2 * count, dtype=np.float64)
    path = tmp_path / module.RHS_RECORD
    path.write_bytes(
        b"NEMO_L1_RHS___1".ljust(16)
        + struct.pack("=7i", 1, module.STEP, 3, 3, 2, 2, 64)
        + fields.tobytes()
    )
    result = module.read_completed_rhs(path)
    assert result["header"]["kt"] == module.STEP

    changed = bytearray(path.read_bytes())
    changed[16:20] = struct.pack("=i", 2)
    path.write_bytes(changed)
    with pytest.raises(Exception, match="bad header"):
        module.read_completed_rhs(path)


def test_bt_reader_rejects_truncation(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (6, 5, 2))
    count = 6 * 5
    path = tmp_path / module.BT_RECORD
    payload = (
        b"NEMO_L1_BTFRM_1".ljust(16)
        + struct.pack("=6i", 1, module.STEP, 3, 6, 5, 64)
        + np.arange(4 * count, dtype=np.float64).tobytes()
    )
    path.write_bytes(payload)
    result = module.read_bt_handoff(path)
    assert result["uu_b"].shape == (1, 2)
    path.write_bytes(payload[:-8])
    with pytest.raises(Exception, match="malformed"):
        module.read_bt_handoff(path)
