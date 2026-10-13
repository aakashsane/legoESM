import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires"


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script], check=False, text=True,
        capture_output=True, cwd=ROOT,
    )


def test_deck_selector_derives_names_and_evidence_from_deck():
    deck = TOOL / "deck.sh"
    for name, run_name, suffix in (
        ("shipped", "GYRE_OMIP_L2_P3", "oracle_gyre_r1_stage"),
        ("r4", "GYRE_OMIP_L2_P3_R4", "oracle_gyre_r4_stage"),
        ("r4_smag", "GYRE_OMIP_L2_P3_R4_SMAG",
         "oracle_gyre_r4_smag_stage"),
    ):
        result = _bash(
            f"source {deck}; gyre_hires_select_deck {name} {TOOL}; "
            'printf "%s|%s\\n" "$RUN_NAME" "$DEFAULT_EVIDENCE"'
        )
        assert result.returncode == 0, result.stderr
        round_name = "round2" if name == "r4_smag" else "round1"
        assert result.stdout.strip().split("|") == [
            run_name,
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
            f"gyrehr_rounds/{round_name}/{suffix}",
        ]


def test_r4_patch_resolves_exact_operator_physics_and_stage_clock(tmp_path):
    target = tmp_path / "namelist_cfg"
    shutil.copy(TOOL.parent / "configs/gyre_omip_l2_phase3_namelist_cfg", target)
    with (TOOL / "namelist_cfg_r4_kt10.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(target)], stdin=patch,
            check=False, capture_output=True,
        )
    assert result.returncode == 0, result.stderr.decode()
    checked = _bash(
        f"source {TOOL / 'deck.sh'}; gyre_hires_verify_deck r4 {target}"
    )
    assert checked.returncode == 0, checked.stderr


def test_r4_smag_patch_is_exact_operator_launcher_block(tmp_path):
    target = tmp_path / "namelist_cfg"
    shutil.copy(TOOL.parent / "configs/gyre_omip_l2_phase3_namelist_cfg", target)
    with (TOOL / "namelist_cfg_r4_smag_kt10.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(target)], stdin=patch,
            check=False, capture_output=True,
        )
    assert result.returncode == 0, result.stderr.decode()
    checked = _bash(
        f"source {TOOL / 'deck.sh'}; gyre_hires_verify_deck r4_smag {target}"
    )
    assert checked.returncode == 0, checked.stderr
    text = target.read_text()
    block = text[text.index("&namdyn_ldf"):text.index("/", text.index("&namdyn_ldf"))]
    assert "rn_Uv" not in block and "rn_Lv" not in block


def test_selector_refuses_unknown_deck():
    result = _bash(
        f"source {TOOL / 'deck.sh'}; gyre_hires_select_deck unknown {TOOL}"
    )
    assert result.returncode == 64
    assert "REFUSE_DECK" in result.stderr


def test_ahm_writer_patch_only_changes_the_two_p3_steppers(tmp_path):
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
        "cfgs/GYRE_OMIP_L2_P3/MY_SRC"
    )
    if not source.is_dir():
        import pytest
        pytest.skip("local NEMO P3 writer source is unavailable")
    for name in ("stprk3.F90", "stprk3_stg.F90"):
        shutil.copy(source / name, tmp_path / name)
    with (TOOL / "gyre_p3_ahm_writer.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", "-d", str(tmp_path), "-p1"], stdin=patch,
            check=False, capture_output=True,
        )
    assert result.returncode == 0, result.stderr.decode()
    assert "NEMO_L4_AHMTF__1" in (tmp_path / "stprk3.F90").read_text()
    stage = (tmp_path / "stprk3_stg.F90").read_text()
    assert "NEMO_L4_DYNLDF_1" in stage
    assert stage.index("l4_uu_before(:,:,:) =") < stage.index("CALL dyn_ldf")
    assert stage.index("CALL dyn_ldf") < stage.index("WRITE(l4_unit)")


def test_ahm_builder_read_only_preflight_applies_pinned_patch():
    result = subprocess.run(
        ["bash", str(TOOL / "build_ahm_writer.sh")],
        check=False, text=True, capture_output=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "PREFLIGHT_OK" in result.stdout
    assert "GYRE_OMIP_L2_P3_AHM" in result.stdout
