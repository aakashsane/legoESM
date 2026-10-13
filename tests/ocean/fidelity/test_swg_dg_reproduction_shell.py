"""Portable preflight tests for the committed NEMO SWG_DG shell tools."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/data/nemo_swg_double_gyre"
MOUNTED_NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")


@pytest.mark.parametrize("name", (
    "build_swg_dg.sh", "check_swg_dg_identity.sh", "swg_md_run.sh",
    "swg_dg_rankcheck.sh",
))
def test_tool_help_and_source_have_no_personal_path_defaults(name):
    path = TOOLS / name
    result = subprocess.run(
        ["bash", str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    source = path.read_text()
    assert "/data/" not in source
    assert "/home/" not in source


def test_reference_arch_keeps_the_measured_scalar_math_flags():
    source = (TOOLS / "arch-conda-scalarmath.fcm.in").read_text()
    assert source.count("@CONDA_PREFIX@") == 2  # prose plus the one value token
    assert "%CPP                 cpp -Dkey_nosignedzero" in source
    assert ("-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer "
            "-ffree-line-length-none -fallow-argument-mismatch "
            "-fno-tree-vectorize") in source
    assert "%CFLAGS              -O0" in source


@pytest.mark.skipif(
    not (MOUNTED_NEMO / "tests/SWG_DG/BLD/bin/nemo.exe").is_file(),
    reason="built SWG_DG case is not mounted")
def test_nemo_run_and_controls_preflight_without_mpi_or_makenemo(tmp_path):
    cfg = MOUNTED_NEMO / "tests/SWG_DG"
    run = subprocess.run([
        "bash", str(TOOLS / "swg_md_run.sh"),
        "--nemo-cfg", str(cfg), "--output-root", str(tmp_path),
        "--three-year", "portable", "32",
    ], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "steps=1892160" in run.stdout
    assert "nn_stock=630720" in run.stdout
    assert run.stdout.endswith("PREFLIGHT_OK\n")
    assert not (tmp_path / "portable").exists()

    identity = subprocess.run([
        "bash", str(TOOLS / "check_swg_dg_identity.sh"),
        "--nemo-root", str(MOUNTED_NEMO),
        "--arch-name", "conda-scalarmath",
        "--output", str(tmp_path / "identity"),
    ], capture_output=True, text=True)
    assert identity.returncode == 0, identity.stderr
    assert "steps=10 ranks=1" in identity.stdout
    assert identity.stdout.endswith("PREFLIGHT_OK\n")

    rank = subprocess.run([
        "bash", str(TOOLS / "swg_dg_rankcheck.sh"),
        "--nemo-cfg", str(cfg), "--output", str(tmp_path / "rank"),
    ], capture_output=True, text=True)
    assert rank.returncode == 0, rank.stderr
    assert "rank-control=1-versus-32" in rank.stdout
    assert rank.stdout.endswith("PREFLIGHT_OK\n")
