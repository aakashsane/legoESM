import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_climate.py"
SPEC = importlib.util.spec_from_file_location("gyre_hires_climate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLIMATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIMATE)


def test_domain_assembly_uses_one_based_xy_positions_and_refuses_overlap():
    tiles = [
        (np.array([1, 1]), np.array([[1.0, 2.0], [3.0, 4.0]])),
        (np.array([3, 1]), np.array([[5.0], [6.0]])),
    ]
    assert np.array_equal(
        CLIMATE.assemble_domain_tiles(tiles, shape=(2, 3)),
        np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]),
    )
    with pytest.raises(Exception, match="overlapping"):
        CLIMATE.assemble_domain_tiles(tiles + [(np.array([1, 1]), np.ones((2, 1)))],
                                      shape=(2, 3))


def test_weighted_sst_is_divided_by_same_file_thickness():
    weighted = np.array([[20.0, 42.0]])
    thickness = np.array([[2.0, 3.0]])
    assert np.array_equal(
        CLIMATE.decode_weighted_surface_temperature(weighted, thickness),
        np.array([[10.0, 14.0]]),
    )
    with pytest.raises(Exception, match="thickness"):
        CLIMATE.decode_weighted_surface_temperature(weighted, np.zeros_like(thickness))


def test_diagnose_applies_31_day_highpass_and_registered_statistics():
    shape = (6, 8)
    wet = np.ones(shape, dtype=bool)
    geom = {
        "wet": wet,
        "u_mask": wet.copy(),
        "v_mask": wet.copy(),
        "area": np.ones(shape),
        "latitude": np.broadcast_to(np.arange(shape[0])[:, None], shape),
        "f": np.ones(shape),
        "spacing_m": 1.0,
    }
    frames = []
    for day in range(1, 36):
        phase = np.sin(2.0 * np.pi * day / 7.0)
        ssh = np.broadcast_to((np.arange(shape[0]) - 2.5)[:, None], shape).copy()
        frames.append(CLIMATE.DailyFrame(
            day, ssh, np.full(shape, 2.0 + phase), np.full(shape, phase),
            np.full(shape, 10.0 + day),
        ))
    provider = CLIMATE.DailyProvider("synthetic", "daily_mean", {}, lambda: iter(frames))
    result = CLIMATE.diagnose(provider, target_start=16, target_days=20, geom=geom)
    assert result["basin_mean_ke_series_m2_s2"].shape == (20,)
    assert result["highpass_eke_map_m2_s2"].shape == shape
    assert float(result["eddy_fraction"]) > 0.0
    assert float(result["ssh_zero_crossing_row"]) == 2.5


def test_preregistered_score_has_two_relative_and_two_row_bars():
    base = {
        "window_mean_basin_ke_m2_s2": np.asarray(1.0),
        "eddy_fraction": np.asarray(0.1),
        "absolute_u_jet_row": np.asarray(10),
        "ssh_zero_crossing_row": np.asarray(11.0),
    }
    report = CLIMATE.score_statistics({"NEMO": base, "legoESM": dict(base)})
    assert report["verdict"] == "PASS"
    assert [row["bar"] for row in report["rows"]] == [0.2, 0.2, 2.0, 2.0]
