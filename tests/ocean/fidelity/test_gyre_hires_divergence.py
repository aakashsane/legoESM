import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_divergence.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_hires_divergence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIVERGENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIVERGENCE)


def test_normalized_rms_and_correlation_known_answer_and_one_ulp_plant():
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    candidate = reference + 1.0
    mask = np.ones_like(reference, dtype=bool)
    row = DIVERGENCE.normalized_rms_and_correlation(
        reference, candidate, mask)
    assert row["normalized_rms_difference"] == pytest.approx(1.0 / np.std(reference))
    assert row["pattern_correlation"] == pytest.approx(1.0)

    exact = DIVERGENCE.normalized_rms_and_correlation(reference, reference, mask)
    planted = reference.copy()
    planted[0, 0] = np.nextafter(planted[0, 0], np.inf)
    moved = DIVERGENCE.normalized_rms_and_correlation(reference, planted, mask)
    assert exact["normalized_rms_difference"] == 0.0
    assert moved["normalized_rms_difference"] > 0.0


def test_divergence_curve_preserves_daily_order():
    reference = np.stack([
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[2.0, 4.0], [6.0, 8.0]]),
    ])
    candidate = reference.copy()
    candidate[1, 0, 0] += 1.0
    curve = DIVERGENCE.divergence_curve(
        reference, candidate, np.ones((2, 2), dtype=bool))
    assert np.array_equal(curve["normalized_rms_difference"][:1], [0.0])
    assert curve["normalized_rms_difference"][1] > 0.0


def test_yearly_eddy_fraction_uses_full_series_and_all_years():
    day = np.arange(730, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * day / 7.0)[:, None, None]
    u = np.broadcast_to(2.0 + signal, (730, 2, 3)).copy()
    v = np.broadcast_to(signal, (730, 2, 3)).copy()
    rows = DIVERGENCE.yearly_eddy_statistics(
        u, v, np.ones((2, 3), dtype=bool))
    assert [row["year"] for row in rows] == [1, 2]
    assert all(row["eddy_fraction"] > 0.0 for row in rows)
    zero_eddy = DIVERGENCE.yearly_eddy_statistics(
        np.full_like(u, 2.0), np.zeros_like(v), np.ones((2, 3), dtype=bool))
    assert [row["eddy_fraction"] for row in zero_eddy] == [0.0, 0.0]


def test_twin_benchmark_refuses_a_shifted_curve():
    curve = {
        "ssh": {"normalized_rms_difference": np.zeros(365)},
        "u": {"normalized_rms_difference": np.zeros(365)},
    }
    for day, value in DIVERGENCE.OPERATOR_TWIN_SSH_BENCHMARK.items():
        curve["ssh"]["normalized_rms_difference"][day - 1] = float(value)
    assert DIVERGENCE._validate_twin(curve)["status"] == "PASS"
    curve["ssh"]["normalized_rms_difference"][29] *= 2.0
    with pytest.raises(Exception, match="benchmark"):
        DIVERGENCE._validate_twin(curve)
