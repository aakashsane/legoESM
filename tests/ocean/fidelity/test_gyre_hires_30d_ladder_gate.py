import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_30d_ladder_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_hires_30d_ladder_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_daily_schedule_covers_initial_and_30_complete_days():
    schedule = GATE.day_boundary_schedule()
    assert len(schedule) == 31
    assert schedule[:3] == (1, 25, 49)
    assert schedule[-1] == 721
    assert np.all(np.diff(schedule) == 24)


def test_daily_schedule_refuses_empty_duration():
    with pytest.raises(ValueError, match="positive"):
        GATE.day_boundary_schedule(0)


def test_daily_gate_preregisters_exact_bar_and_acquisition_root():
    assert "round6/oracle_gyre_r4_smag_daily_stage/run" in str(
        GATE.DEFAULT_ORACLE_ROOT)
    assert GATE.STEPS_PER_DAY == 24 and GATE.NDAYS == 30


def test_daily_vector_oracle_admission_checks_all_boundaries():
    admission = GATE.admit_daily_oracle(GATE.DEFAULT_ORACLE_ROOT, "vector")
    entries = [name for name in admission["checked_files"]
               if name.startswith("oracle_step_entry_")]
    assert len(entries) == 31
    assert admission["math_flavour"] == "vector"


def test_daily_oracle_refuses_unregistered_scalar_flavour():
    with pytest.raises(ValueError, match="unregistered daily oracle"):
        GATE.admit_daily_oracle(GATE.DEFAULT_ORACLE_ROOT, "scalar")
