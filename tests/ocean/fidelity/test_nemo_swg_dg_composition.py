"""Direct tests for the SWG_DG program-composition discriminator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = (Path(__file__).resolve().parents[3]
        / "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l1_swg_dg_composition.py")
SPEC = importlib.util.spec_from_file_location("swg_dg_composition", PATH)
composition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(composition)


def test_field_totals_separates_prognostic_and_forcing_signed_zero_debt():
    rows = [
        {"field": "sshn", "bit_identical": True, "bit_unequal": 0,
         "numeric_unequal": 0, "signed_zero_only": 0},
        {"field": "vtau_b", "bit_identical": False, "bit_unequal": 640,
         "numeric_unequal": 0, "signed_zero_only": 640},
    ]
    totals = composition._field_totals(rows)
    assert totals["prognostic_bit_identical"]
    assert not totals["all_bit_identical"]
    assert totals["bit_unequal"] == totals["signed_zero_only"] == 640


def test_cli_requires_explicit_root_and_json():
    with pytest.raises(SystemExit) as error:
        composition.main([])
    assert error.value.code == 2
