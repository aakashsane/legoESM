"""The AIMIP lanes freeze the option-unread knobs by name (#1945 follow-up).

``aimip_inactive_fields`` = the cloud-scheme set plus ``resolved_inactive_fields``
of the config the AIMIP factory builds.  Every lane that builds the
spec-driven bundle (run_aimip trainer, WB scale_build, run_aimip_wb2_eval
skeleton) must route through it, or a Morrison arm trains k_au/k_ac/x_star
that kk2000 never reads (run_aimip's no-inert gate then aborts the run).
Each lane test fails if its call is reverted to the cloud-only set.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm.training import aimip_params as ap

ROOT = Path(__file__).resolve().parents[2]
_SB = {"atm.micro.MorrisonConfig." + f for f in ("k_au", "k_ac", "x_star")}
_KK = {"atm.micro.MorrisonConfig." + f
       for f in ("autocon_fact", "accre_enhan_fact")}
_HTOPO = "atm.gwd.McFarlaneConfig.h_topo"
_PIN_SB = {"atm.micro.MorrisonConfig": {"warm_rain_scheme": "seifert_beheng"}}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_helper_follows_scheme_pins_and_orography():
    kw = dict(cloud_scheme="cam6_clubb", microphysics="morrison",
              gwd="mcfarlane")
    cloud = ap._INACTIVE_FIELDS_BY_CLOUD_SCHEME["cam6_clubb"]
    assert ap.aimip_inactive_fields(**kw, subgrid_orography=False) == (
        cloud | _SB)
    assert ap.aimip_inactive_fields(**kw, subgrid_orography=True) == (
        cloud | _SB | {_HTOPO})
    assert ap.aimip_inactive_fields(**kw, subgrid_orography=False,
                                    param_fixed=_PIN_SB) == cloud | _KK
    assert ap.aimip_inactive_fields(
        cloud_scheme="xu_randall", microphysics="sundqvist", gwd="none",
        subgrid_orography=True) == frozenset()


def _run_aimip_bundle(monkeypatch):
    import legoesm.training.neural_gcm_spectral as ngs
    mod = _load("run_aimip_lane_inactive", "scripts/run/run_aimip.py")
    cfg = mod._merge(mod._load_yaml(ROOT / "config" / "aimip" / "aimip_era5.yaml"), {
        "n_max": 8, "aimip_variant": "classical", "aimip_spatial_surface": False,
        "aimip_radiation": "gray", "aimip_rad_update_interval": 1,
        "aimip_microphysics": "morrison", "aimip_trainable_schemes": "extended"})
    spec_cfg = mod._build_spectral_config(cfg)

    def fake_data(spec_cfg_, grid, sigma, cache_dir, **kw):
        carry = types.SimpleNamespace(phis=jnp.zeros((grid.n_lat, grid.n_lon)))
        return [carry], [carry], [0]
    seen = {}
    monkeypatch.setattr(ngs, "load_training_data", fake_data)
    monkeypatch.setattr(ngs, "_train_spectral_loop",
                        lambda params, *a, **k: seen.update(params=params, kw=k))
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: (lambda *a2, **k2: None))
    mod._train_aimip_classical(spec_cfg, "unused", cfg=cfg)
    return seen, cfg, spec_cfg


def test_run_aimip_and_eval_skeleton_freeze_unread_morrison_knobs(monkeypatch):
    seen, cfg, spec_cfg = _run_aimip_bundle(monkeypatch)
    assert seen["kw"].get("assert_no_inert_params") is True
    bundle = seen["params"]
    assert isinstance(bundle, ap.AIMIPTrainableBundle)
    names = set(bundle.schemes.raw_values)
    assert not (names & _SB), sorted(names & _SB)      # kk2000 never reads them
    assert _KK <= names                                  # kk2000 does
    # the eval skeleton must hold the SAME tree, or the checkpoint will not load
    from legoesm.grids.gaussian import create_gaussian_grid
    ev = _load("run_aimip_wb2_eval_lane_inactive",
               "scripts/validate/run_aimip_wb2_eval.py")
    skel = ev._build_skeleton("classical", cfg, spec_cfg, create_gaussian_grid(
        spec_cfg.n_max, dealiasing="quadratic"))
    assert set(skel.schemes.raw_values) == names
    assert (jax.tree.structure(skel.schemes)
            == jax.tree.structure(bundle.schemes))


def test_scale_build_freezes_by_resolved_warm_rain_law(monkeypatch):
    from legoesm.training import scale_build
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: None)
    cfg = types.SimpleNamespace(mode="physics", training_core="spectral",
                                smoke=True, multi_step_hours=(6,), n_days=1)

    def names(param_fixed):
        yml = {"era5_zarr": "synthetic", "era5_surface_fluxes": False,
               "train_years": [2019], "radiation": "gray",
               "n_lat": 8, "n_lon": 16, "nlev": 4,
               "spectral": {"n_max": 5, "dt": 60.0}, "loss": {},
               "classical": {"microphysics": "morrison",
                             "trainable_schemes": "extended",
                             "param_fixed": param_fixed}}
        params = scale_build.build_mode_components(cfg, yml)[3]
        return set(params.schemes.raw_values)

    kk = names({})
    assert not (kk & _SB) and _KK <= kk
    sb = names(_PIN_SB)                    # the deck pins the SB law
    assert not (sb & _KK) and _SB <= sb
