"""MorrisonConfig.prognostic_droplet_number, stage S10.3: MG2 cloud-droplet
sedimentation (mass, number, evaporation of the share falling into clear air,
surface flux) and the droplet-number clear, against the RUNNING MG2.

Fixture ``oracle_fixtures/mg2_prognostic_nc_sed.npz``: ``make_fixture.py
--prognostic-nc-sed`` through the probe build ``mg2_tap`` (build.sh: a copy of
micro_mg2_0.F90 of cam_cesm2_1_rel_60 with 8 added lines that only record
nctend after the activation re-apply, nctend after sedimentation (before
:2501) and the droplet sub-step count).  All columns are warm and ice-free, so
MG2's whole-call qctend, qcsedten, qvlat, tlat and surface fluxes contain no
process this port has not reached yet.  Bound 1e-10 relative, x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_sed.npz"


def _fx():
    return dict(np.load(FIXTURE))


def _rel_flux(a, b, gross):
    """Relative error against max(|value|, gross): where the cloud-water limiter
    binds, the sedimenting state q_c + dq_c*dt = q_c*(1 - omsm) is a 1e-5
    residue of q_c, so its absolute round-off is eps*q_c, not eps*residue."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.asarray(gross))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def test_fixture_inputs():
    fx = _fx()
    assert np.allclose(fx["col_rho"], fx["col_p_full"] / (C.R_d * fx["col_T"]), rtol=1e-14)
    wet = (fx["col_q_c"] > 0) | (fx["col_q_r"] > 0)
    assert (fx["col_T"][wet] > C.T_freeze).all()          # warm water: no unported cold process


def _gross(fx, out):
    """Outgoing term at the START-of-step water, fc(post state) |q_c| / pdel."""
    dt = float(fx["dt"])
    rho = jnp.asarray(fx["col_rho"])
    q0 = jnp.asarray(fx["col_q_c"])
    q = q0 + (out.dq_c_dt - out.qc_budget["sedimentation"]) * dt
    fc, _ = W.mg2_droplet_fall_coefficients(
        q, jnp.asarray(fx["col_nc_kg"]), jnp.maximum(jnp.asarray(fx["col_cf"]), 1e-4),
        jnp.asarray(fx["col_T"]), rho)
    return np.abs(np.asarray(q0 * fc)) / np.diff(fx["col_p_half"], axis=1)


def test_sedimentation_matches_mg2():
    fx = _fx()
    out = A._run(fx)
    dt = float(fx["dt"])
    rho = fx["col_rho"]
    qb = out.qc_budget
    nc = fx["col_nc_kg"]
    errs = {
        "qctend": _rel(out.dq_c_dt, fx["ora_qctend"]),
        "qcsedten": _rel_flux(qb["sedimentation"], fx["ora_qcsedten"], _gross(fx, out)),
        "qvlat": _rel(out.dq_v_dt, fx["ora_qvlat"]),
        "tlat": _rel(np.asarray(out.dT_dt) * C.c_pd, fx["ora_tlat"]),
        "precip": _rel(out.precipitation,
                       fx["ora_rflx_bot"][:, -1] + fx["ora_lflx_bot"][:, -1]),
    }
    clear = fx["col_q_c"] + fx["ora_qctend"] * dt < 1e-18
    neg = ~clear & (nc + fx["ora_nctend_sed"] * dt < 0.0)
    keep = ~clear & ~neg
    dN = np.asarray(out.dN_c_dt) / rho
    errs["nctend_sed"] = _rel(dN[keep], fx["ora_nctend_sed"][keep])
    errs["nctend_clear"] = _rel(dN[clear], fx["ora_nctend"][clear])
    worst = {k: float(v.max()) for k, v in errs.items()}
    print("S10.3 max rel err", worst, "cells", int(keep.sum()), "clear", int(clear.sum()),
          "floor", int(neg.sum()))
    assert max(worst.values()) <= TOL, worst
    assert np.allclose(dN[neg], -nc[neg] / dt, rtol=1e-12, atol=0.0)   # interim floor
    # coverage, on the MG2 side
    names = list(fx["col_names"])
    thin = names.index("thin")
    assert fx["ora_nstepc"][thin, 0] > 1                       # sub-stepping active
    cf = fx["col_cf"][thin]
    shrink = np.r_[False, (cf[1:] < cf[:-1]) & (fx["col_q_c"][thin][:-1] > 0)]
    assert (fx["ora_qvlat"][thin][shrink] > 0).any()           # inflow evaporates (no rain here)
    assert (fx["col_q_r"][thin] == 0).all()
    clr = names.index("clearnc")
    assert (clear[clr] & (nc[clr] > 0)).any()                  # number clear with droplets left
    assert np.abs(fx["ora_qcsedten"]).max() > 0
    # the sub-step count is reported with the other species'
    assert int(np.asarray(out.sed_substeps_required)[thin]) >= int(fx["ora_nstepc"][thin, 0])


def test_column_water_closes():
    fx = _fx()
    out = A._run(fx)
    pdel = np.diff(fx["col_p_half"], axis=1)
    col = ((np.asarray(out.dq_v_dt) + np.asarray(out.dq_c_dt) + np.asarray(out.dq_r_dt))
           * pdel / C.g).sum(axis=1) + np.asarray(out.precipitation)
    scale = (np.abs(np.asarray(out.dq_c_dt)) * pdel / C.g).sum(axis=1).max()
    assert np.abs(col).max() <= 1e-12 * scale


def test_gradients_through_sedimentation():
    fx = _fx()
    rho = jnp.asarray(fx["col_rho"])
    thin = list(fx["col_names"]).index("thin")

    def f(nc, qc):
        out = A._run(fx, N_c=nc, q_c=qc)
        return jnp.sum(out.qc_budget["sedimentation"][thin]) * 1e6 + jnp.sum(out.dq_v_dt) * 1e6

    g = jax.grad(f, argnums=(0, 1))(jnp.asarray(fx["col_nc_kg"]) * rho, jnp.asarray(fx["col_q_c"]))
    for a in g:
        assert np.isfinite(np.asarray(a)).all()
    assert np.abs(np.asarray(g[0])[thin]).max() > 0 and np.abs(np.asarray(g[1])[thin]).max() > 0


def test_switch_cap_is_the_shared_one():
    cfg = _cfg()
    assert cfg.sed_cfl_substeps and int(cfg.sed_cfl_substeps_max) >= int(_fx()["ora_nstepc"].max())
