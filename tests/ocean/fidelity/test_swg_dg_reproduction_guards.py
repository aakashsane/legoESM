"""Direct tests for the cross-deck and short-ladder reproduction guards."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/validate/ocean_fidelity/testcases"
FIXTURE = ROOT / "tests/ocean/fixtures/swg_dg/output.namelist.dyn"
PACKAGE = ROOT / "scripts/data/nemo_swg_double_gyre"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deck = _load("swg_dg_deck_guard_test",
             TOOLS / "nemo_testcase_l1_swg_dg_deck_guard.py")
driver = _load("swg_dg_driver_guard_test",
               ROOT / "scripts/run/run_swg_double_gyre.py")
short = _load("swg_dg_short_ladder_test",
              TOOLS / "nemo_testcase_l1_swg_dg_short_ladder.py")


def test_resolved_nemo_and_legoesm_decks_differ_only_in_named_io_controls():
    nemo = deck.parse_resolved_echo(FIXTURE)
    result = deck.compare_resolved(nemo, driver.resolved_daily_namelist(1095))
    assert result["overlap_compared"] == 137
    assert [row["field"] for row in result["differences"]] == [
        "sn_cfctl.l_runstat", "nn_stock"]
    assert all(row["allowlisted"] and row["reason"]
               for row in result["differences"])


def test_cross_deck_guard_rejects_one_changed_physical_value():
    nemo = deck.parse_resolved_echo(FIXTURE)
    nemo["rn_tau"] = 0.11
    with pytest.raises(RuntimeError, match="rn_tau_pa"):
        deck.compare_resolved(nemo, driver.resolved_daily_namelist(1095))


def test_short_ladder_scores_every_step_and_field_and_localizes_plant():
    masks = {name: np.ones((2, 2), dtype=bool) for name in "TUV"}
    restarts = {
        step: {name: np.full((2, 2), float(step))
               for name in ("sshn", "un", "vn")}
        for step in range(1, 11)
    }
    trajectory = [
        {name: np.full((2, 2), float(step))
         for name in ("ssh_m", "u_m_s", "v_m_s")}
        for step in range(1, 11)
    ]

    def exact(reference, candidate, mask, *, name):
        unequal = int(np.count_nonzero(
            np.asarray(reference).view(np.uint64)[mask]
            != np.asarray(candidate).view(np.uint64)[mask]))
        return {"name": name, "bit_identical": unequal == 0,
                "bit_unequal": unequal, "max_abs": float(
                    np.max(np.abs(reference[mask] - candidate[mask])))}

    rows = short.score_trajectory(
        restarts, trajectory, masks, exact_row=exact)
    assert len(rows) == 30
    assert all(row["bit_identical"] for row in rows)
    trajectory[3]["v_m_s"][1, 0] = np.nextafter(4.0, np.inf)
    rows = short.score_trajectory(
        restarts, trajectory, masks, exact_row=exact)
    debts = [row for row in rows if not row["bit_identical"]]
    assert [(row["step"], row["field"], row["bit_unequal"])
            for row in debts] == [(4, "v_m_s", 1)]


def test_environment_reconstruction_files_are_machine_readable_and_pinned():
    explicit = (PACKAGE / "nemo-build-linux-64-explicit.txt").read_text()
    assert "# platform: linux-64" in explicit
    assert "@EXPLICIT" in explicit
    urls = [line for line in explicit.splitlines() if line.startswith("https://")]
    assert len(urls) > 100
    assert all("conda.anaconda.org/conda-forge/" in url for url in urls)
    assert any("openmpi-5.0.10-h67ed482_1" in url for url in urls)
    assert any("netcdf-fortran-4.6.3-mpi_openmpi_h8f6c4f5_0" in url
               for url in urls)

    readable = yaml.safe_load(
        (PACKAGE / "nemo-build-environment.yml").read_text())
    assert readable["name"] == "swg-dg-nemo-build"
    assert "openmpi=5.0.10" in readable["dependencies"]
    assert "netcdf-fortran=4.6.3" in readable["dependencies"]

    requirements = [
        Requirement(line) for line in
        (PACKAGE / "python-gpu-requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    pinned = {item.name.lower(): str(item.specifier) for item in requirements}
    assert pinned["jax"] == "==0.10.0"
    assert pinned["jax-cuda12-plugin"] == "==0.10.0"
    assert pinned["nvidia-cuda-runtime-cu12"] == "==12.9.79"
    assert pinned["numpy"] == "==2.4.4"

    environment = json.loads((
        ROOT / "docs/ocean/fidelity/testcases/"
        "swg_double_gyre_environment.json").read_text())
    assert environment["nemo"]["repository"] == (
        "https://forge.nemo-ocean.eu/nemo/nemo.git")
    assert environment["hardware"]["minimum_gpu_memory_bytes"] == 32 * 2**30
