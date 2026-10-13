import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/run/run_gyre_hires.py"
SPEC = importlib.util.spec_from_file_location("run_gyre_hires", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def test_gpu_selection_is_explicit_exact_and_cpu_forbids_it():
    DRIVER.validate_physical_gpu("gpu-native", 0, {"CUDA_VISIBLE_DEVICES": "0"})
    DRIVER.validate_physical_gpu("gpu-native", 3, {"CUDA_VISIBLE_DEVICES": "3"})
    DRIVER.validate_physical_gpu("cpu-certified", None, {})
    with pytest.raises(ValueError, match="requires --physical-gpu"):
        DRIVER.validate_physical_gpu("gpu-native", None, {})
    with pytest.raises(RuntimeError, match="must equal"):
        DRIVER.validate_physical_gpu("gpu-native", 1, {"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="does not accept"):
        DRIVER.validate_physical_gpu("cpu-certified", 1, {})


def test_gpu_wrapper_has_no_default_device():
    wrapper = ROOT / "scripts/run/run_gyre_hires_gpu.sh"
    result = subprocess.run(
        ["bash", str(wrapper)], check=False, text=True, capture_output=True,
    )
    assert result.returncode == 64
    assert "--physical-gpu PHYSICAL_INDEX" in result.stderr


def test_resolved_daily_decks_keep_declared_resolution_and_cadence():
    shipped = DRIVER.resolved_daily_namelist("shipped", 2)
    r4 = DRIVER.resolved_daily_namelist("r4", 1095)
    smag = DRIVER.resolved_daily_namelist("r4_smag", 1095)
    assert (shipped.nn_GYRE, shipped.rn_Dt_s, shipped.nn_itend, shipped.nn_write) == (
        1,
        14400.0,
        12,
        6,
    )
    assert (r4.nn_GYRE, r4.rn_Dt_s, r4.nn_itend, r4.nn_write, r4.nn_stock) == (
        4,
        3600.0,
        26280,
        24,
        8760,
    )
    assert (smag.nn_GYRE, smag.ln_dynldf_blp, smag.nn_ahm_ijk_t, smag.rn_Uv_m_s, smag.rn_Lv_m) == (
        4,
        True,
        32,
        None,
        None,
    )
    with pytest.raises(ValueError, match="deck must"):
        DRIVER.resolved_daily_namelist("invented", 1)


@pytest.mark.slow
def test_tiny_real_gyre_two_day_scan_matches_certified_loop():
    import jax
    from legoesm.core.precision import get_policy, set_policy

    if jax.default_backend() != "cpu":
        pytest.skip("real two-day parity is a CPU control")
    old = get_policy()
    try:
        namelist = DRIVER.resolved_daily_namelist("shipped", 2)
        report = DRIVER.cpu_scan_parity(namelist, days=2)
    finally:
        set_policy(old)
    assert report["days"] == 2
    assert report["steps"] == 12
    assert report["differences"]["all_finite"] is True
    assert report["forcing_transcription"]["clock_and_phase_bit_equal"] is True
    assert report["classification"] == "COMPOSITION"
    assert report["prefix_comparisons"][0]["model_step"] == 2
    assert report["prefix_comparisons"][0]["fields"] == []
    assert report["prefix_comparisons"][0]["differences"]["all_bit_equal"] is True
    assert report["prefix_comparisons"][1]["model_step"] == 3
    assert report["first_difference"]["step"] == 3
    assert report["first_difference"]["first_field"] == "ssh_m"
    # Measured refutation: enclosing the same production step in a day scan
    # changes XLA's fused floating-point program.  Keep the failed 1e-12 bar
    # visible rather than silently relaxing it after seeing the residual.
    assert report["differences"]["within_atol"] is False
    assert report["differences"]["surface_temperature_c_max_abs"] == (
        pytest.approx(1.8106049992638873e-10, rel=1.0e-6)
    )
