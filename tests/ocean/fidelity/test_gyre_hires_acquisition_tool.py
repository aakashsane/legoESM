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
