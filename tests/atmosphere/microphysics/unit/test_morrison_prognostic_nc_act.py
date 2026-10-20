"""MorrisonConfig.prognostic_droplet_number, stage S10.2: the MG2 droplet-number
budget (activation pre-apply and re-apply, nprc1/npra/riming number sinks from
the unlimited mass rates, the nc limiter with omsm), against the RUNNING MG2.

Fixture ``oracle_fixtures/mg2_prognostic_nc_act.npz``: ``make_fixture.py
--prognostic-nc-act`` through the probe build ``mg2_tap`` (build.sh: a copy of
micro_mg2_0.F90 of cam_cesm2_1_rel_60 with three added lines recording nctend
right after the activation re-apply, micro_mg2_0.F90:2022).  Bound: 1e-10
relative, hard-switch limit.  Where MG2's number would end the step negative
(N_c + npccn*dt < 0), MG2 leaves the repair to its end of step (later stages);
there the interim net floor applies (user decision 2026-10-20) and is asserted.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_act.npz"
OMSM = 1.0 - 1.0e-5


def _fx():
    return dict(np.load(FIXTURE))


def _run(fx, cfg=None, npccn_kg=None, N_c=None, q_c=None, hyd_kw=None, T=None):
    g = {k: jnp.asarray(fx[f"col_{k}"]) for k in
         ("T", "q_v", "q_c", "cf", "q_r", "N_r", "p_full", "p_half", "rho", "nc_kg", "relvar",
          "npccn")}
    dz = (g["p_half"][:, 1:] - g["p_half"][:, :-1]) / (C.g * g["rho"])
    z = jnp.zeros_like(g["T"])
    kw = dict(q_c=g["q_c"] if q_c is None else q_c, q_r=g["q_r"], q_i=z, q_s=z, q_g=z,
              N_c=g["nc_kg"] * g["rho"] if N_c is None else N_c, N_r=g["N_r"], N_i=z)
    kw.update(hyd_kw or {})
    npc = g["npccn"] if npccn_kg is None else npccn_kg
    return morrison_microphysics(
        g["T"] if T is None else T, g["q_v"], HydrometeorState(**kw), g["p_full"],
        g["p_half"], g["rho"], dz, float(fx["dt"]), cfg or _cfg(),
        cloud_fraction=g["cf"], relvar=g["relvar"], npccn=npc * g["rho"])


def _mg2_masks(fx):
    nc, npc, qc = fx["col_nc_kg"], fx["col_npccn"], fx["col_q_c"]
    dt = float(fx["dt"])
    pre = np.where(qc >= 1e-18, np.maximum(nc + npc * dt, 0.0), nc)
    fin = nc + fx["ora_nctend_stage"] * dt
    neg = fin < 0.0
    bound = (pre > 0) & (np.abs(fin / np.where(pre > 0, pre, 1.0) - (1.0 - OMSM)) < 1e-6)
    return pre, neg, bound


def test_droplet_number_budget_matches_mg2_stage():
    fx = _fx()
    out = _run(fx)
    dN = np.asarray(out.dN_c_dt) / fx["col_rho"]          # per kg, as MG2
    _, neg, bound = _mg2_masks(fx)
    err = _rel(dN[~neg], fx["ora_nctend_stage"][~neg])
    print("S10.2 nctend max rel err", float(err.max()), "cells", int((~neg).sum()),
          "floor cells", int(neg.sum()))
    assert float(err.max()) <= TOL
    # interim floor (user decision 2026-10-20) where MG2's number would go negative
    floor = -fx["col_nc_kg"] / float(fx["dt"])
    assert np.allclose(dN[neg], floor[neg], rtol=1e-12, atol=0.0)
    qb = out.qc_budget
    for k, ora in (("autoconversion", "ora_prctot"), ("accretion", "ora_pratot")):
        e = float(_rel(-np.asarray(qb[k]), fx[ora]).max())
        assert e <= TOL, (k, e)
    # coverage, all on the MG2 side
    npc, qc = fx["col_npccn"], fx["col_q_c"]
    assert bound.any()                                           # number limiter binds
    witness = (fx["ora_qcrat"] < 1.0) & ~bound & (npc == 0) & (qc > 0)
    assert witness.any()                                         # mass limited, number not
    assert ((npc > 0) & (qc > 0) & ~neg).any() and ((npc < 0) & (qc > 0) & ~neg).any()
    assert ((npc != 0) & (qc == 0) & ~neg).any()                 # re-apply in clear air
    assert neg.sum() < (~neg).sum()


def test_activation_pre_apply_moves_the_rates():
    fx = _fx()
    with_act = np.asarray(_run(fx).qc_budget["autoconversion"])
    without = np.asarray(_run(fx, npccn_kg=jnp.zeros_like(jnp.asarray(fx["col_npccn"])))
                         .qc_budget["autoconversion"])
    moved = np.abs(with_act - without) > 1e-6 * np.abs(without)
    assert (moved & (fx["col_npccn"] != 0)).any()
    assert not (moved & (fx["col_npccn"] == 0)).any()


def test_cold_riming_number_sinks_and_no_ice_riming():
    """Cold cloudy cells with snow, graupel and ice: riming by cloud ice is 0;
    the droplet-number sink is (snow + graupel riming) x the size-reset ncic/qcic
    where autoconversion and accretion vanish (q_r = 0, qcic < icsmall)."""
    fx = _fx()
    qc0 = jnp.asarray(fx["col_q_c"])
    cloudy = qc0 > 0
    cf = jnp.maximum(jnp.asarray(fx["col_cf"]), 1e-4)
    q_c = jnp.where(cloudy, 5.0e-9 * cf, 0.0)                    # qcic = 5e-9 < 1e-8
    T = jnp.where(cloudy, 258.0, jnp.asarray(fx["col_T"]))
    sol = jnp.where(cloudy, 1.0e-4, 0.0)
    cfg = _cfg()
    out = _run(fx, cfg=cfg, npccn_kg=jnp.zeros_like(qc0), q_c=q_c, T=T,
               hyd_kw=dict(q_r=jnp.zeros_like(qc0), q_s=sol, q_g=sol, q_i=0.1 * sol,
                           N_i=jnp.where(cloudy, 1.0e5, 0.0)))
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    assert np.all(qb["riming_ice"] == 0.0)
    rim = -(qb["riming_snow"] + qb["riming_graupel"])
    c = np.asarray(cloudy)
    assert (rim[c] > 0).all() and np.all(qb["autoconversion"][c] == 0) \
        and np.all(qb["accretion"][c] == 0)
    sinks = -sum(v for k, v in qb.items() if k != "condensation")
    dt = float(fx["dt"])
    assert (sinks[c] * dt < 0.99 * np.asarray(q_c)[c]).all()   # qc mass limiter not binding
    rho = jnp.asarray(fx["col_rho"])
    qcic = q_c / cf
    ncic = W.mg2_size_bounded_nc(qcic, jnp.asarray(fx["col_nc_kg"]) * rho / cf, rho)
    want = -rim * np.asarray(ncic / qcic)
    got = np.asarray(out.dN_c_dt)
    nc_vol = np.asarray(jnp.asarray(fx["col_nc_kg"]) * rho)
    has_n = c & (nc_vol > 0)
    assert (-want[has_n] * dt < nc_vol[has_n]).all()            # nc limiter not binding
    assert float(_rel(got[has_n], want[has_n]).max()) <= 1e-12
    # a cloudy cell with no droplets: the nc limiter removes nothing (MG2 :1628-1642)
    assert (c & (nc_vol == 0)).any() and np.all(got[c & (nc_vol == 0)] == 0.0)


def test_npccn_guards():
    fx = _fx()
    g = {k: jnp.asarray(fx[f"col_{k}"]) for k in
         ("T", "q_v", "q_c", "cf", "q_r", "N_r", "p_full", "p_half", "rho", "nc_kg", "relvar")}
    dz = (g["p_half"][:, 1:] - g["p_half"][:, :-1]) / (C.g * g["rho"])
    z = jnp.zeros_like(g["T"])
    hyd = HydrometeorState(q_c=g["q_c"], q_r=g["q_r"], q_i=z, q_s=z, q_g=z,
                           N_c=g["nc_kg"] * g["rho"], N_r=g["N_r"], N_i=z)
    args = (g["T"], g["q_v"], hyd, g["p_full"], g["p_half"], g["rho"], dz, float(fx["dt"]))
    with pytest.raises(ValueError, match="npccn= with shape"):
        morrison_microphysics(*args, _cfg(), cloud_fraction=g["cf"], relvar=g["relvar"])
    with pytest.raises(ValueError, match="npccn= with shape"):
        morrison_microphysics(*args, _cfg(), cloud_fraction=g["cf"], relvar=g["relvar"],
                              npccn=jnp.asarray(0.0))
    with pytest.raises(ValueError, match="read only under"):
        morrison_microphysics(*args, _cfg()._replace(prognostic_droplet_number=False),
                              cloud_fraction=g["cf"], npccn=z)


def test_gradients_through_activation():
    fx = _fx()
    _, neg, _ = _mg2_masks(fx)
    keep = jnp.asarray(~neg)
    rho = jnp.asarray(fx["col_rho"])

    def f(npc, nc, qc):
        out = _run(fx, npccn_kg=npc, N_c=nc, q_c=qc)
        return jnp.sum(jnp.where(keep, out.dN_c_dt, 0.0)) * 1e-6 + jnp.sum(out.dq_r_dt)

    g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(fx["col_npccn"]),
                                         jnp.asarray(fx["col_nc_kg"]) * rho,
                                         jnp.asarray(fx["col_q_c"]))
    for a in g:
        assert np.isfinite(np.asarray(a)).all()
    assert np.abs(np.asarray(g[0])[~neg]).max() > 0.0
