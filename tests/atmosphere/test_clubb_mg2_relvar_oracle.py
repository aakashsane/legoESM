"""CAM6 CLUBB -> MG2 relative cloud-water variance (relvar) vs the RUNNING oracles.

Fixture ``oracle_fixtures/clubb_mg2_relvar.npz``, generator
``scripts/validate/oracle/clubb_mg2_relvar/run.sh`` (metadata in the npz):

* Part A: on the Fortran CLUBB-core trajectories of ``clubb_core_cam.npz``
  (regenerated bit for bit, with the Fortran post-advance ``pdf_params``),
  ``relvar_oracle.F90`` evaluates the oracle's lines for ``rcp2`` (CLUBB
  pdf_closure_module.F90 r8099 :1048-1050, :1072-1074) and ``qclvar`` /
  ``relvar`` (CAM clubb_intr.F90 cam_cesm2_1_rel_60 :1998, :2416-2425,
  deep_scheme = 'ZM').
* Part B: the six tier-1 warm columns through CAM cam_cesm2_1_rel_60
  ``micro_mg_tend`` (compiled unmodified) with a per-level relvar profile.

Tolerance 1e-10 in fp64 (user decision 2026-10-19).  ``rcp2`` uses the CLUBB
core test's field-normalised metric on the same trajectories; relvar and the
MG2 outputs use the MG2 test's element-wise ``|a-b|/max(|a|,|b|)``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/atmosphere/oracle_fixtures/clubb_mg2_relvar.npz"
TOL = 1.0e-10
CASES = ["bomex", "dycoms2_rf01", "arm", "wangara", "mpace_b", "bomex_dz100",
         "arm_dz160", "bomex_dryut", "arm_dryut"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = _load(ROOT / "tests/atmosphere/test_clubb_core_oracle.py", "clubb_core_oracle")
mg = _load(ROOT / "tests/atmosphere/microphysics/unit/test_mg2_warm_rain_oracle.py",
           "mg2_oracle")


@pytest.fixture(scope="module")
def rv():
    return dict(np.load(FIX))


@pytest.fixture(scope="module")
def cfx():
    return dict(np.load(core._FIX))


def _ours(cfx, case):
    """Our (rcm, rcp2_zt) after each teacher-forced step s_n -> s_{n+1}."""
    gr, forcing, env = core._inputs(cfx, case)
    out = []
    for n in range(int(cfx["meta/n_steps"])):
        st, rcm = core._state(cfx, case, n)
        o = core._step(st, rcm, forcing, env, gr, core._CFG, extra=("rcp2_zt",))
        out.append((np.asarray(o["rcm"]), np.asarray(o["rcp2_zt"])))
    return out


def test_fixture_provenance(rv):
    meta = json.loads(str(rv["meta"]))
    assert meta["cam_sha"] == "a03b84b7c4e34f965b115686f22a043b85739e56"
    assert "54403a04390fc4a0d4865357556e77c1ed6ad513" in meta["clubb_oracle"]
    assert [str(c) for c in rv["meta/cases"]] == CASES


@pytest.mark.parametrize("case", CASES)
def test_rcp2_matches_fortran(rv, cfx, case):
    worst = 0.0
    for n, (_, rcp2) in enumerate(_ours(cfx, case), start=1):
        ref = rv[f"{case}/s{n}/rcp2"]
        assert np.all(np.isfinite(rcp2))
        worst = max(worst, float(np.max(np.abs(rcp2[0] - ref))
                                 / max(np.max(np.abs(ref)), 1e-300)))
    assert worst <= TOL, (case, worst)


def test_rcp2_fixture_is_not_trivial(rv):
    """Cloudy and clear levels both present; rcp2 > 0 somewhere in every
    cloudy case."""
    pos = [np.count_nonzero(rv[f"{c}/s1/rcp2"]) for c in CASES]
    zero = [np.count_nonzero(rv[f"{c}/s1/rcp2"] == 0) for c in CASES]
    assert sum(pos) > 50 and sum(zero) > 50, (pos, zero)


def _rel(a, b):
    return mg._rel(a, b)


def test_relvar_formula_matches_oracle(rv, cfx):
    """cam6_relvar on the oracle's own rcm/rcp2 = clubb_intr's relvar, incl.
    every branch (default, both clips, qclvar cap, rcm == 0, rcp2 == 0)."""
    rcm, rcp2, ref = [], [], []
    for c in CASES:
        for n in range(1, int(rv["meta/n_steps"]) + 1):
            rcm.append(cfx[f"{c}/s{n}/rcm"][0])
            rcp2.append(rv[f"{c}/s{n}/rcp2"])
            ref.append(rv[f"{c}/s{n}/relvar"])
    e = rv["edge/in"]
    rcm = np.concatenate(rcm + [e[:, -1]])
    rcp2 = np.concatenate(rcp2 + [rv["edge/rcp2"]])
    ref = np.concatenate(ref + [rv["edge/relvar"]])
    ours = np.asarray(C.cam6_relvar(jnp.asarray(rcm), jnp.asarray(rcp2)))
    assert float(_rel(ours, ref).max()) <= TOL
    edge = rv["edge/relvar"]
    assert np.sum(edge == 10.0) >= 3 and np.any(edge == 1.0e-3)
    assert np.any((edge > 1.0e-3) & (edge < 10.0))
    assert np.any(rv["edge/rcp2"] > 1.0)          # the qclvar cap binds


@pytest.mark.parametrize("case", CASES)
def test_relvar_end_to_end(rv, cfx, case):
    """Our rcm and rcp2 from our CLUBB step -> relvar, vs the oracle's.

    Field-normalised, the CLUBB core certificate's metric: relvar inherits
    rcp2, a difference of terms that cancels by up to 4 orders at trace cloud
    (worst element 5.6e-10, wangara s3, rcp2 = 2.5e-16 against a column max
    of 1.3e-12), so its element-wise error is the core's 1e-12 times that
    cancellation, not a closure defect.
    """
    for n, (rcm, rcp2) in enumerate(_ours(cfx, case), start=1):
        ours = np.asarray(C.cam6_relvar(jnp.asarray(rcm), jnp.asarray(rcp2)))[0]
        ref = rv[f"{case}/s{n}/relvar"]
        err = float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))
        assert err <= TOL, (case, n, err)


def _mg2_cfg():
    cfg = mg._production_morrison()
    assert cfg.warm_rain_scheme == "kk2000_cam6"      # the deck's selection
    return cfg._replace(**mg.HARD_LIMIT)


def _mg2_errors(rv, relvar):
    mfx = mg._fx()
    out = mg._run_columns(mfx, _mg2_cfg(), relvar=relvar)
    qb = out.qc_budget
    evap = np.asarray(out.dq_v_dt) + np.asarray(qb["condensation"])
    errs = {
        "prc": _rel(-np.asarray(qb["autoconversion"]), rv["mg2/prctot"]),
        "pra": _rel(-np.asarray(qb["accretion"]), rv["mg2/pratot"]),
        "evap": _rel(evap, rv["mg2/nevapr"]),
        "dq_r": _rel(out.dq_r_dt, rv["mg2/qrtend"]),
        "dN_r": _rel(np.asarray(out.dN_r_dt) / mfx["col_rho"], rv["mg2/nrtend"]),
        "precip": _rel(out.precipitation, rv["mg2/rain_sfc_flux"]),
    }
    return {k: float(v.max()) for k, v in errs.items()}


def test_mg2_with_per_level_relvar_matches_micro_mg_tend(rv):
    mfx = mg._fx()
    relvar = jnp.broadcast_to(jnp.asarray(rv["mg2/relvar"]), mfx["col_T"].shape)
    worst = _mg2_errors(rv, relvar)
    assert max(worst.values()) <= TOL, worst


def test_mg2_per_level_relvar_is_not_the_constant(rv):
    """Non-vacuity: the deck's constant fallback (no per-cell relvar) misses
    the oracle on the same columns."""
    worst = _mg2_errors(rv, None)
    assert max(worst.values()) > 1.0e3 * TOL, worst


def test_gradients_finite():
    rcm = jnp.asarray([0.0, 1e-6, 2e-4, 3e-3, -4e-5])
    rcp2 = jnp.asarray([1e-8, 1e-6, 2e-8, 1e-8, 1e-8])
    g = jax.grad(lambda a, b: jnp.sum(C.cam6_relvar(a, b)), argnums=(0, 1))(rcm, rcp2)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)
    assert float(jnp.max(jnp.abs(g[0]))) > 0.0
    mfx = mg._fx()
    cfg = _mg2_cfg()

    def loss(r):
        out = mg._run_columns(mfx, cfg, relvar=jnp.broadcast_to(r, mfx["col_T"].shape))
        return jnp.sum(out.precipitation)

    gr = jax.grad(loss)(jnp.full(mfx["col_T"].shape[1], 2.0))
    assert bool(jnp.all(jnp.isfinite(gr))) and float(jnp.max(jnp.abs(gr))) > 0.0


def test_deck_routes_clubb_relvar_to_mg2():
    """The production deck selects the per-cell path: kk2000_cam6 with the
    CLUBB liquid partition (the condition make_physics keys the hand-off on)."""
    from scripts.run.run_amip import build_arg_parser, build_config_from_args

    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    keys = load_yaml_config(mg.DECK, parser, example_keys="'microphysics'")
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    cfg = build_config_from_args(parser.parse_args(["--config", str(mg.DECK)]))
    assert cfg.morrison_warm_rain_scheme == "kk2000_cam6"
    assert cfg._liquid_partition_resolved()
    assert cfg.cld_macmic_num_steps >= 2


@pytest.mark.parametrize("incloud", [True, False])
def test_microphysics_reads_clubbs_relvar_from_the_same_substep(monkeypatch, incloud):
    """MPAS lane, CAM order: the relvar Morrison receives in macmic sub-step i
    is the one prognostic CLUBB returned in sub-step i."""
    import sys
    sys.path.insert(0, str(ROOT / "tests/unit"))
    from test_physics_macmic import _moist_setup

    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics import integration as mi
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence import integration as ti
    written, read = [], []
    real_upd = ti._carry_update_with_cloud_fraction
    real_micro = mi.morrison_microphysics

    def upd(carry_field, carry_val, turb_out):
        written.append(np.asarray(turb_out.relvar))
        return real_upd(carry_field, carry_val, turb_out)

    def micro(*a, relvar=None, **k):
        read.append(None if relvar is None else np.asarray(relvar))
        return real_micro(*a, relvar=relvar, **k)
    monkeypatch.setattr(ti, "_carry_update_with_cloud_fraction", upd)
    monkeypatch.setattr(mi, "morrison_microphysics", micro)
    mesh, sigma, state = _moist_setup()
    # drop the setup's warm bump so the lower column is supersaturated and
    # CLUBB makes liquid (else relvar is relvarmax everywhere)
    sig = jnp.asarray(sigma.sigma_full)[None, :]
    state = state._replace(T=state.T.replace(data=state.T.data - 20.0 * sig ** 8))
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none",
                                  cloud_config=CloudConfig(scheme="cam6_clubb")),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", liquid_partition=True,
                                    clubb=C.CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(warm_rain_scheme="kk2000_cam6",
                                    warm_rain_incloud=incloud)))
    fn = make_physics(cfg, model_type="mpas", dt=600.0, cld_macmic_num_steps=2)
    ps = init_physics_state(*state.T.data.shape, cfg)
    fn(state, mesh, sigma, phys_state=ps)
    assert len(written) == len(read) == 2
    for w, r in zip(written, read):
        assert r is not None
        np.testing.assert_array_equal(w.reshape(r.shape), r)
    # discriminating: per-cell values, not the pbuf seed or the constant
    assert np.any((read[0] > 1.0e-3) & (read[0] < 10.0)), np.unique(read[0])
    assert not np.all(read[0] == 2.0)
    # sub-steps differ, so equality per sub-step pins the ordering
    assert not np.array_equal(written[0], written[1])
