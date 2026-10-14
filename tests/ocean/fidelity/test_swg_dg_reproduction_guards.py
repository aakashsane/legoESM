"""Direct tests for the cross-deck and short-ladder reproduction guards."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/validate/ocean_fidelity/testcases"
FIXTURE = ROOT / "tests/ocean/fixtures/swg_dg/output.namelist.dyn"
NAMBDY_FIXTURE = ROOT / "tests/ocean/fixtures/swg_dg/namelist_ref.nambdy"
PACKAGE = ROOT / "scripts/data/nemo_swg_double_gyre"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deck = _load("swg_dg_deck_guard_test",
             TOOLS / "nemo_testcase_l1_swg_dg_deck_guard.py")
short = _load("swg_dg_short_ladder_test",
              TOOLS / "nemo_testcase_l1_swg_dg_short_ladder.py")


def _fixture_run(tmp_path: Path, *, reference: str | None = None) -> Path:
    run = tmp_path / "nemo_run"
    run.mkdir()
    (run / "output.namelist.dyn").write_bytes(FIXTURE.read_bytes())
    (run / "namelist_ref").write_text(
        NAMBDY_FIXTURE.read_text() if reference is None else reference)
    (run / "namelist_cfg").write_text("&nambdy\n/\n")
    (run / "ocean.output").write_text(
        "NEMO completed; nambdy values are not printed by this executable\n")
    hashes = []
    for name in ("namelist_ref", "namelist_cfg"):
        digest = hashlib.sha256((run / name).read_bytes()).hexdigest()
        hashes.append(f"{digest}  {name}\n")
    (run / "manifest.sha256").write_text("".join(hashes))
    return run


def test_resolved_nemo_and_legoesm_decks_cover_all_compiled_reads(tmp_path):
    result = deck.run(_fixture_run(tmp_path), 1095)
    assert result["compiled_inventory_fields_resolved"] == 131
    assert result["compiled_inventory_fields_required"] == 131
    assert result["comparison_rows"] == 158
    assert result["case_specific_echo_rows"] == 16
    assert [row["field"] for row in result["differences"]] == [
        "sn_cfctl.l_runstat", "nn_stock"]
    assert all(row["allowlisted"] and row["reason"]
               for row in result["differences"])
    assert result["all_physical_and_numerical_values_equal"]
    assert result["nemo_sources"]["ocean_output_cross_checks"]["nambdy"] == {
        "status": "not_printed",
        "matched_fields": [],
        "note": (
            "NEMO's ocean.output contains no machine-readable value for this "
            "echo-omitted group; the content-hashed namelist_ref/namelist_cfg "
            "precedence is authoritative"),
    }
    nambdy_sources = {
        source for name, source in result["resolved_sources"].items()
        if name.startswith("nambdy.")}
    assert nambdy_sources == {"namelist_ref"}


def test_cross_deck_guard_rejects_changed_echoed_physical_value(tmp_path):
    run = _fixture_run(tmp_path)
    echo = (run / "output.namelist.dyn").read_text().replace(
        "RN_TAU= 0.10000000000000001", "RN_TAU= 0.11000000000000000")
    (run / "output.namelist.dyn").write_text(echo)
    with pytest.raises(RuntimeError, match="rn_tau_pa"):
        deck.run(run, 1095)


def test_cross_deck_guard_rejects_changed_unechoed_ln_bdy(tmp_path):
    reference = NAMBDY_FIXTURE.read_text().replace(
        "ln_bdy           = .false.", "ln_bdy           = .true.")
    with pytest.raises(RuntimeError, match="ln_bdy"):
        deck.run(_fixture_run(tmp_path, reference=reference), 1095)


def test_cross_deck_guard_refuses_compiled_field_missing_from_every_source(
        tmp_path):
    reference = "\n".join(
        line for line in NAMBDY_FIXTURE.read_text().splitlines()
        if "ln_bdy" not in line) + "\n"
    with pytest.raises(RuntimeError, match="unresolved compiled-read field nambdy.ln_bdy"):
        deck.run(_fixture_run(tmp_path, reference=reference), 1095)


def test_cross_deck_guard_applies_nemo_cfg_over_ref_precedence(tmp_path):
    reference = NAMBDY_FIXTURE.read_text().replace(
        "ln_bdy           = .false.", "ln_bdy           = .true.")
    run = _fixture_run(tmp_path, reference=reference)
    (run / "namelist_cfg").write_text("&nambdy\n ln_bdy = .false.\n/\n")
    digest = hashlib.sha256((run / "namelist_cfg").read_bytes()).hexdigest()
    ref_digest = hashlib.sha256((run / "namelist_ref").read_bytes()).hexdigest()
    (run / "manifest.sha256").write_text(
        f"{ref_digest}  namelist_ref\n{digest}  namelist_cfg\n")
    result = deck.run(run, 1095)
    assert result["resolved_sources"]["nambdy.ln_bdy"] == "namelist_cfg"


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


def test_short_ladder_reference_comparator_localizes_departure(tmp_path, capsys):
    baseline = json.loads(short.REFERENCE_BASELINE.read_text())
    paths = {}
    for composition in ("outer-scan", "materialized"):
        form = baseline["forms"][composition]
        signatures = short._expected_signatures(form)
        rows = []
        for step in range(1, 11):
            for field in ("ssh_m", "u_m_s", "v_m_s"):
                bit, numeric, signed, maximum = signatures[(step, field)]
                rows.append({
                    "step": step, "field": field,
                    "bit_unequal": bit, "numeric_unequal": numeric,
                    "signed_zero_only": signed, "max_abs": maximum,
                })
        payload = {
            "schema": "swg-dg-short-restart-ladder-v1",
            "composition": composition,
            "all_exact": form["all_exact"],
            "rows": rows,
        }
        path = tmp_path / f"{composition}.json"
        path.write_text(json.dumps(payload))
        paths[composition] = path

    args = [
        "--compare-reference", "--outer-report", str(paths["outer-scan"]),
        "--materialized-report", str(paths["materialized"]),
    ]
    assert short.main(args) == 0
    output = capsys.readouterr().out
    assert "outer-scan same as reference" in output
    assert "materialized same as reference" in output

    materialized = json.loads(paths["materialized"].read_text())
    planted = next(row for row in materialized["rows"]
                   if row["step"] == 4 and row["field"] == "v_m_s")
    planted["bit_unequal"] += 1
    paths["materialized"].write_text(json.dumps(materialized))
    assert short.main(args) == 1
    output = capsys.readouterr().out
    assert "materialized departs from reference at step=4 field=v_m_s" in output


def test_environment_reconstruction_files_are_machine_readable_and_pinned():
    explicit = (PACKAGE / "nemo-build-linux-64-explicit.txt").read_text()
    assert "# platform: linux-64" in explicit
    assert "@EXPLICIT" in explicit
    urls = [line for line in explicit.splitlines() if line.startswith("https://")]
    assert len(urls) == 122
    assert all("conda.anaconda.org/conda-forge/" in url for url in urls)
    assert all(re.fullmatch(r"https://.+#[0-9a-f]{32}", url) for url in urls)
    unhashed_urls = [url.rsplit("#", 1)[0] for url in urls]
    assert hashlib.sha256(
        ("\n".join(unhashed_urls) + "\n").encode()).hexdigest() == (
            "deed23744c963e5c5787cd0731517a3aebd6475872a3c8938c3566eb0d80d56f")
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
    pinned = {
        canonicalize_name(item.name): str(item.specifier)
        for item in requirements}
    assert len(pinned) == 40
    assert pinned["jax"] == "==0.10.0"
    assert pinned["jax-cuda12-plugin"] == "==0.10.0"
    assert pinned["nvidia-cuda-runtime-cu12"] == "==12.9.79"
    assert pinned["numpy"] == "==2.4.4"
    assert pinned["typing-extensions"] == "==4.15.0"
    assert pinned["certifi"] == "==2026.4.22"

    pending = list(pinned)
    visited = set()
    missing_required_edges = []
    while pending:
        parent = pending.pop()
        if parent in visited:
            continue
        visited.add(parent)
        for raw in metadata.distribution(parent).requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            child = canonicalize_name(requirement.name)
            if child not in pinned:
                missing_required_edges.append((parent, child, raw))
            else:
                pending.append(child)
    assert visited == set(pinned)
    assert missing_required_edges == []

    environment = json.loads((
        ROOT / "docs/ocean/fidelity/testcases/"
        "swg_double_gyre_environment.json").read_text())
    assert environment["nemo"]["repository"] == (
        "https://forge.nemo-ocean.eu/nemo/nemo.git")
    assert environment["hardware"]["minimum_gpu_memory_bytes"] == 32 * 2**30

    costs = json.loads((
        ROOT / "docs/ocean/fidelity/testcases/"
        "swg_double_gyre_reference_costs.json").read_text())
    assert costs["nemo_three_year"]["wall_seconds"] == 22_918
    assert (costs["legoesm_gpu_three_year_daily_mean"]["wall_seconds"]
            == 8_117)
    assert costs["legoesm_cpu_one_day"]["wall_seconds"] == pytest.approx(
        233.93357684463263)
    assert all(
        len(costs[name]["source_artifact_sha256"]) == 64
        for name in ("nemo_three_year", "legoesm_gpu_three_year_daily_mean",
                     "legoesm_cpu_one_day"))


def test_stranger_runbook_contains_every_reconstruction_and_failure_command():
    runbook = (ROOT / "docs/ocean/fidelity/testcases/"
               "swg_double_gyre_reproduction.md").read_text()
    required = (
        "https://forge.nemo-ocean.eu/nemo/nemo.git",
        "dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796",
        "nemo-build-linux-64-explicit.txt",
        "python-gpu-requirements.txt",
        "python=3.13.0=h9ebbce0_101_cp313",
        "typing-extensions==4.15.0",
        "certifi==2026.4.22",
        "--python \"$PY\"",
        "--short-ladder dg_ladder",
        "--composition outer-scan",
        "--composition materialized",
        "--backend gpu --gpu-device 0 --composition outer-scan",
        "--compare-reference",
        "compiled_reads=131/131",
        "414,545",
        "peak_bytes_in_use",
        "100 GiB",
        "352 passed, 18 skipped",
        "SWG_DG_RESTART_IDENTITY PASS day=365",
        "SWG_DG_MOVIE COMPLETE frames=365",
        "cross-toolchain acceptance bar",
        "2.286230926795696e-11",
        "1.402270660699756e-11",
        "export STAGE_ROOT=/path/to/instrumented/nemo/step2/stage-entry-records",
    )
    for text in required:
        assert text in runbook
    assert "sole source of" not in runbook
    assert "full-restart identity" in runbook
    assert runbook.count("JAX_PLATFORMS=cpu") >= 6
    assert re.findall(r"^## ([1-9])\. ", runbook, re.MULTILINE) == list(
        "123456789")
    battery_files = (
        "tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py",
        "tests/ocean/unit/test_nemo_tsunami_card.py",
        "tests/ocean/fidelity/test_nemo_swg_check_records.py",
        "tests/ocean/fidelity/test_nemo_swg_dg_climate.py",
        "tests/ocean/fidelity/test_nemo_swg_dg_composition.py",
        "tests/ocean/fidelity/test_nemo_swg_dg_restart_identity.py",
        "tests/ocean/fidelity/test_nemo_swg_dg_round6.py",
        "tests/ocean/fidelity/test_nemo_swg_dg_round7.py",
        "tests/ocean/fidelity/test_nemo_swg_geometry_gate.py",
        "tests/ocean/fidelity/test_nemo_swg_ladder.py",
        "tests/ocean/fidelity/test_nemo_tsunami_ladder.py",
        "tests/ocean/fidelity/test_swg_dg_reproduction_guards.py",
        "tests/ocean/fidelity/test_swg_dg_reproduction_shell.py",
        "tests/ocean/fidelity/test_swg_double_gyre_movie.py",
        "tests/ocean/unit/test_nemo_swg_card.py",
        "tests/ocean/unit/test_pv_flux_ene.py",
        "tests/ocean/unit/test_swg_swe_rk3_stage.py",
        "tests/ocean/validation/test_step_gradient_matrix.py",
    )
    assert all(runbook.count(path) == 1 for path in battery_files)
