import importlib.util
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget"
)
PACKAGER_PATH = TOOL / "package_records.py"
SPEC = importlib.util.spec_from_file_location(
    "gyre_hires_budget_package_records", PACKAGER_PATH)
PACKAGER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PACKAGER)


def write_record(path: Path, *, day: int = 30, plant=None) -> None:
    ni, nj = 126, 86
    mask = np.zeros((ni, nj), dtype=np.float64)
    interior_words = np.zeros(122 * 82, dtype=np.float64)
    interior_words[:9480] = 1.0
    interior = interior_words.reshape((122, 82), order="F")
    mask[2:-2, 2:-2] = interior
    terms = np.zeros((ni, nj, len(PACKAGER.TERMS)), dtype=np.float64)
    if plant == "non_finite":
        terms[2, 2, 0] = np.nan
    elif plant == "closure":
        terms[2, 2, 0] = 1.0e-8
    header = (
        PACKAGER.MAGIC
        + struct.pack(
            "=9i", 1, day, day * 24, ni, nj, 30, 64,
            len(PACKAGER.TERMS), 24)
        + struct.pack("=2d", 3600.0, PACKAGER.CLOSURE_BAR_ABS_M_S)
    )
    path.write_bytes(
        header + mask.tobytes(order="F") + terms.tobytes(order="F"))


def write_restart(path: Path, value: float) -> None:
    from netCDF4 import Dataset

    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        variable = dataset.createVariable("v", "f8", ("x",))
        variable[:] = np.asarray([value, 1.0], dtype=np.float64)


def test_record_reader_closes_and_all_active_plants_refuse(tmp_path):
    good = tmp_path / "good.bin"
    write_record(good)
    record = PACKAGER.read_record(good)
    assert record["day"] == 30
    assert record["terms"].shape == (8, 82, 122)
    assert int(np.count_nonzero(record["mask"])) == 9480
    assert record["closure_max_abs_m_s"] == 0.0

    for plant, match in (
        ("non_finite", "non-finite"), ("closure", "closure")):
        path = tmp_path / f"{plant}.bin"
        write_record(path, plant=plant)
        with pytest.raises(ValueError, match=match):
            PACKAGER.read_record(path)

    bad_magic = tmp_path / "bad-magic.bin"
    bad_magic.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="magic"):
        PACKAGER.read_record(bad_magic)


def test_restart_passivity_is_word_exact_and_signed_zero_plant_refuses(tmp_path):
    control = tmp_path / "control.nc"
    same = tmp_path / "same.nc"
    moved = tmp_path / "moved.nc"
    write_restart(control, 0.0)
    write_restart(same, 0.0)
    write_restart(moved, -0.0)
    assert PACKAGER.compare_restarts(control, same)["status"] == "PASS"
    planted = PACKAGER.compare_restarts(control, moved)
    assert planted["status"] == "REFUSE"
    assert planted["unequal_words"] == 1


def test_nemo_patches_apply_exactly_and_scripts_fail_closed(tmp_path):
    nemo = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    sources = {
        "stprk3_stg": nemo / "cfgs/GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90",
        "dynadv": nemo / "src/OCE/DYN/dynadv.F90",
        "dynzdf": nemo / "src/OCE/DYN/dynzdf.F90",
    }
    for stem, source in sources.items():
        target = tmp_path / f"{stem}.F90"
        target.write_bytes(source.read_bytes())
        result = subprocess.run(
            ["patch", "-l", "-s", "--fuzz=0", str(target)],
            input=(TOOL / f"{stem}_round14.patch").read_text(),
            text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "r14_" in target.read_text()

    run = (TOOL / "run.sh").read_text()
    assert "RANKS must be 1" in run
    assert "-d 'OCE'" in run
    assert "-m conda" in run
    assert "conda-scalarmath" not in run
    assert "export GYRE_R14_BUDGET=$enabled" in run
    assert "run_nemo control 0" in run
    assert "run_nemo instrumented 1" in run
    assert "rg " not in run
    assert "mpirun -np 1" in run
    assert "--run" in run
    assert subprocess.run(
        ["bash", "-n", str(TOOL / "run.sh")], check=False).returncode == 0


def test_nemo_writer_has_independent_total_and_no_stress_companion():
    module = (TOOL / "gyre_r14_budget.F90").read_text()
    zdf = (TOOL / "dynzdf_round14.patch").read_text()
    assert "pv_final(:,:) - zr14_entry(:,:)" in module
    assert "pv_no_stress(:,1) - ( zr14_entry(:,kj) + prdt" in module
    assert "zr14_stress(:,kj) = pv_full(:,1) - pv_no_stress(:,1)" in module
    assert "zr14_vnostress" in zdf
    assert "CALL r14_zdf_row( jj, rDt" in zdf
    # The control would be vacuous if it reused the already-solved production
    # V field; it must run a second forward/back substitution from the saved
    # pre-stress RHS.
    assert "zr14_vnostress(ji,jk) = zr14_vnostress(ji,jk) -" in zdf
    assert "pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) -" in zdf
