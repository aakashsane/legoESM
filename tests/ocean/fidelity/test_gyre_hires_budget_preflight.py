import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_budget_preflight", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def write_archive(root: Path, *, drop: str | None = None, non_finite: str | None = None):
    fields = list(PREFLIGHT.LEGO_FULL_FIELDS)
    manifest = {
        "status": "complete",
        "completed_days": 180,
        "per_step": True,
        "full_state_window": [1, 180],
        "full_state_fields": fields,
        "namelist": {"nn_GYRE": 4, "ln_dynldf_blp": True},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for day in (1, 30, 180):
        arrays = {}
        for key in PREFLIGHT.required_legoesm_keys():
            if key != drop:
                arrays[key] = np.array([np.nan if key == non_finite else day])
        np.savez(root / f"day{day:04d}.npz", **arrays)


def test_legoesm_archive_passes_only_with_every_registered_field(tmp_path):
    write_archive(tmp_path)
    report = PREFLIGHT.audit_legoesm_archive(tmp_path)
    assert report["status"] == "PASS"
    assert report["errors"] == []

    missing = tmp_path / "missing"
    missing.mkdir()
    write_archive(missing, drop="v_m_s_3d_daily_mean")
    report = PREFLIGHT.audit_legoesm_archive(missing)
    assert report["status"] == "REFUSE"
    assert any("v_m_s_3d_daily_mean" in error for error in report["errors"])


def test_legoesm_archive_refuses_non_finite_active_provider(tmp_path):
    write_archive(tmp_path, non_finite="temperature_c_3d_daily_mean")
    report = PREFLIGHT.audit_legoesm_archive(tmp_path)
    assert report["status"] == "REFUSE"
    assert any("non-finite temperature_c_3d_daily_mean" in error
               for error in report["errors"])


def test_joint_report_cannot_pass_one_refusing_provider(tmp_path, monkeypatch):
    write_archive(tmp_path)
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {"root": str(root), "status": "REFUSE", "errors": ["plant"]},
    )
    report = PREFLIGHT.build_report(tmp_path, tmp_path / "nemo")
    assert report["verdict"] == "ACQUISITION_NEEDED"
    assert "do not admit" in report["disposition"]


def test_complete_state_archives_do_not_admit_unobserved_production_terms(
        tmp_path, monkeypatch):
    """This is the round-13 regression: full daily state is not a budget."""
    lego = tmp_path / "lego"
    lego.mkdir()
    write_archive(lego)
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {
            "root": str(root), "status": "PASS", "errors": [],
            "online_tendency_variables": [],
            "trend_flags": {flag: False for flag in PREFLIGHT.NEMO_TREND_FLAGS},
        },
    )
    report = PREFLIGHT.build_report(lego, tmp_path / "nemo")
    assert report["state_archive_status"] == "PASS"
    assert report["budget_observability"]["status"] == "REFUSE"
    assert report["owner"] == report["closure"] == "UNMEASURED"
    assert report["verdict"] == "ACQUISITION_NEEDED"


def test_online_contract_can_pass_and_each_semantic_plant_refuses(
        tmp_path, monkeypatch):
    roots = {provider: tmp_path / provider for provider in ("legoesm", "nemo")}
    for provider, root in roots.items():
        root.mkdir()
        shape = (151, 1, 1)
        terms = {
            key: np.zeros(shape) for key in PREFLIGHT.BUDGET_TERM_KEYS}
        np.savez(
            root / "terms.npz",
            day=np.arange(30, 181),
            surface_v_wet_mask=np.ones((1, 1), dtype=bool),
            **terms,
        )
        (root / "passive.json").write_text(json.dumps({
            "provider": provider, "status": "PASS", "unequal_words": 0,
        }))
        (root / "preregistration.json").write_text(json.dumps({
            "terms": list(PREFLIGHT.REGISTERED_TERMS),
            "closure_bar_abs_m_s": 0.0,
            "registered_before_data": True,
            "registration_commit": "a" * 40,
        }))
        contract = {
            "provider": provider,
            **PREFLIGHT.BUDGET_CONTRACT_REQUIREMENTS,
            "closure_bar_abs_m_s": 0.0,
            "data_artifact": "terms.npz",
            "data_sha256": PREFLIGHT.sha256(root / "terms.npz"),
            "passive_identity_artifact": "passive.json",
            "passive_identity_sha256": PREFLIGHT.sha256(root / "passive.json"),
            "preregistration_artifact": "preregistration.json",
            "preregistration_sha256": PREFLIGHT.sha256(
                root / "preregistration.json"),
        }
        (root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(contract))

    monkeypatch.setattr(
        PREFLIGHT, "audit_legoesm_archive",
        lambda root: {"root": str(root), "status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {"root": str(root), "status": "PASS", "errors": []},
    )
    assert PREFLIGHT.build_report(roots["legoesm"], roots["nemo"])["verdict"] == "PASS"

    nemo_contract = json.loads(
        (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).read_text())
    for key in (
        "status", "window_days", "terms", "production_online",
        "external_mode_included",
        "implicit_zdf_stress_separated", "closure_total_recorded",
        "passive_trajectory_identity",
    ):
        planted = dict(nemo_contract)
        planted[key] = False if isinstance(planted[key], bool) else None
        (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
            json.dumps(planted))
        assert PREFLIGHT.build_report(
            roots["legoesm"], roots["nemo"])["verdict"] == "ACQUISITION_NEEDED"

    for key, bad in (
        ("provider", "legoesm"),
        ("closure_bar_abs_m_s", None),
        ("data_sha256", "0" * 64),
        ("passive_identity_sha256", "0" * 64),
        ("preregistration_sha256", "0" * 64),
    ):
        planted = dict(nemo_contract)
        planted[key] = bad
        (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
            json.dumps(planted))
        assert PREFLIGHT.build_report(
            roots["legoesm"], roots["nemo"])["verdict"] == "ACQUISITION_NEEDED"
    (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(nemo_contract))

    # A correctly hashed but non-closing active term must still refuse.
    nemo_root = roots["nemo"]
    arrays = {
        key: np.zeros((151, 1, 1)) for key in PREFLIGHT.BUDGET_TERM_KEYS}
    arrays[PREFLIGHT.BUDGET_TERM_KEYS[0]][0, 0, 0] = 1.0
    np.savez(
        nemo_root / "terms.npz", day=np.arange(30, 181),
        surface_v_wet_mask=np.ones((1, 1), dtype=bool), **arrays)
    planted = dict(nemo_contract)
    planted["data_sha256"] = PREFLIGHT.sha256(nemo_root / "terms.npz")
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"


def test_daily_means_cannot_identify_a_nonlinear_daily_mean_product():
    control = PREFLIGHT.nonlinear_daily_mean_control()
    assert control["provider_means_a"] == control["provider_means_b"] == [0.0, 0.0]
    assert control["mean_product_a"] == 1.0
    assert control["mean_product_b"] == -1.0
    assert control["ambiguity"] == 2.0
    assert control["control_fires"] is True
