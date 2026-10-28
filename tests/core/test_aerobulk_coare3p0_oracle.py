"""Oracle fixture: ``compute_most_fluxes(scheme="coare3")`` reproduces the
COMPILED AeroBulk 1.0.0 ``TURB_COARE3P0``.

Oracle: brodeau/aerobulk tag 1.0.0 (SHA ca40928872eb9c00f96a51cf3a6eed0bb4a70580),
mod_blk_coare3p0 / mod_common_coare / mod_phymbl compiled unmodified, mod_const
with legoESM's grav and R_vap (still default-real literals, as AeroBulk writes
every constant), no cool-skin / warm-layer, nb_iter = 5.  The fixture
``oracle_fixtures/aerobulk_coare3p0.npz`` holds the oracle's inputs and outputs
for 1549 states (ERA5 open ocean incl. seven named regimes, the frozen
aerobulk-python baseline, ten synthetic edges) at zt/zu = 10/10 and 2/10 m; its
``meta`` string records the provenance.  Regenerate with::

    bash scripts/validate/oracle/aerobulk/build.sh <aerobulk-1.0.0> <build>
    python scripts/validate/oracle/aerobulk/make_fixture.py --drv <build>/drv \\
        --scheme coare3p0 --ic <ic_2001.nc> --fc <forcing_2001.nc> \\
        --out tests/core/oracle_fixtures/aerobulk_coare3p0.npz

The oracle fluxes are AeroBulk's BULK_FORMULA structure (tau = rho Cd Ub |U|,
Q_H = rho cp Ch Ub (t_zu - T_s), E = rho Ce Ub (q_zu - q_s)) built from its
Cd, Ch, Ce, Ub, t_zu, q_zu with the model's rho, cp = c_pd and L; AeroBulk heat
fluxes are positive DOWN and its stress lies along the wind, so the model's are
compared with a sign flip.  The Charnock coefficient is passed as AeroBulk's
literal 0.011 in single precision (the oracle's value).  Pass: max relative
error <= 1e-10 (fp64), exact agreement where the oracle value is exactly zero.

RULE 5 (user decision 2026-10-28): the neutral SIGN switch of psi_m/h_coare is a
logistic blend of width ``coare_neutral_width`` (default 1e-3), and the public
Charnock default stays 0.011 (fp64).  The oracle match above is the hard limit
(width 0, fp32 Charnock).  At the full default (width 1e-3, Charnock 0.011) the
signed per-state gap (ours - oracle) of every output is pinned against
``oracle_fixtures/aerobulk_coare3p0_default_gap.npz`` (1e-6 relative, 1e-12
absolute).  Regenerate it after a change to the smooth form, its width or the
Charnock default with ``python tests/core/test_aerobulk_coare3p0_oracle.py``.
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402
from legoesm.core.bulk_flux import COARE_NEUTRAL_WIDTH, compute_most_fluxes  # noqa: E402

from legoesm import constants  # noqa: E402

_FIXTURE = Path(__file__).parent / "oracle_fixtures" / "aerobulk_coare3p0.npz"
_GAP_FILE = Path(__file__).parent / "oracle_fixtures" / "aerobulk_coare3p0_default_gap.npz"
_RTOL = 1e-10
_HEIGHTS = {"z10": 10.0, "z2": 2.0}
CHARNOCK_AEROBULK = float(np.float32(0.011))  # AeroBulk charn_coare3p0 literal


def _fixture():
    return np.load(_FIXTURE)


def _inputs(d, h, dtype=jnp.float64):
    return {k: jnp.asarray(d[f"{h}_in_{k}"], dtype=dtype)
            for k in ("u", "v", "th_a", "q", "th_s", "qs", "rho", "L")}


def _ours(x, zt, charnock=CHARNOCK_AEROBULK, **kw):
    tx, ty, sh, lh, us = compute_most_fluxes(
        x["u"], x["v"], x["th_a"], x["q"], x["th_s"], x["qs"], x["rho"],
        z_ref=10.0, z_t=zt, z_q=zt, scheme="coare3", n_iter=5, charnock=charnock,
        L_latent=x["L"], **kw)
    return {"taux": tx, "tauy": ty, "sen": sh, "lat": lh, "ustar": us}


def _oracle(d, h):
    g = lambda k: d[f"{h}_{k}"]  # noqa: E731
    rho, cd, ub = g("in_rho"), g("oracle_Cd"), g("oracle_Ub")
    return {"taux": -rho * cd * ub * g("in_u"), "tauy": -rho * cd * ub * g("in_v"),
            "sen": -rho * constants.c_pd * g("oracle_Ch") * ub * (g("oracle_t_zu") - g("in_th_s")),
            "lat": -rho * g("in_L") * g("oracle_Ce") * ub * (g("oracle_q_zu") - g("in_qs")),
            "ustar": g("oracle_us")}


def max_rel_errors(d, h, **kw):
    ours, ref = _ours(_inputs(d, h), _HEIGHTS[h], **kw), _oracle(d, h)
    out = {}
    for k, b in ref.items():
        a = np.asarray(ours[k])
        zero = b == 0.0
        assert np.all(a[zero] == 0.0), (h, k)
        out[k] = float(np.max(np.abs(a[~zero] - b[~zero]) / np.abs(b[~zero])))
    return out


def test_fixture_provenance_and_coverage():
    d = _fixture()
    meta = str(d["meta"])
    assert "ca40928872eb9c00f96a51cf3a6eed0bb4a70580" in meta and "TURB_COARE3P0" in meta
    for h in _HEIGHTS:
        dt = d[f"{h}_in_th_a"] - d[f"{h}_in_th_s"]
        assert (dt > 0).sum() >= 50 and (dt < 0).sum() >= 50  # both stabilities
        assert np.any(d[f"{h}_in_W"] == 0.0) and np.any(d[f"{h}_in_W"] > 33.0)


@pytest.mark.parametrize("h", sorted(_HEIGHTS))
def test_coare3_matches_compiled_aerobulk_turb_coare3p0(h):
    worst = max_rel_errors(_fixture(), h, coare_neutral_width=0.0)
    bad = {k: v for k, v in worst.items() if not v <= _RTOL}
    assert not bad, (h, worst)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_coare3_gradient_finite_on_fixture(dtype):
    d = _fixture()
    for h, zt in _HEIGHTS.items():
        x = _inputs(d, h, dtype)

        def loss(u, v, th_a, q, th_s, qs):
            y = x | dict(u=u, v=v, th_a=th_a, q=q, th_s=th_s, qs=qs)
            r = _ours(y, zt)
            return jnp.sum(r["taux"] + r["tauy"] + r["sen"] + r["lat"] + r["ustar"])

        args = [x[k] for k in ("u", "v", "th_a", "q", "th_s", "qs")]
        for g in jax.grad(loss, argnums=tuple(range(6)))(*args):
            assert np.all(np.isfinite(np.asarray(g))), (h, dtype)


def test_coare3_jit_matches_eager():
    d = _fixture()
    x = _inputs(d, "z2")
    eager = _ours(x, 2.0)
    jitted = jax.jit(lambda y: _ours(y, 2.0))(x)
    for k in eager:
        np.testing.assert_allclose(np.asarray(jitted[k]), np.asarray(eager[k]),
                                   rtol=_RTOL, atol=0.0, err_msg=k)


@pytest.mark.parametrize("h", sorted(_HEIGHTS))
def test_coare3_2m_diagnostic_closes_at_measurement_height(h):
    """return_2m uses the ported solver's profile (AeroBulk's height shift from
    z_t): at z_diag = z_t it returns the air temperature (codex review: a first
    version kept the legacy 1/L and +-10 clip and returned 273.3 K for 281 K)."""
    d = _fixture()
    x, zt = _inputs(d, h), _HEIGHTS[h]
    out = compute_most_fluxes(x["u"], x["v"], x["th_a"], x["q"], x["th_s"], x["qs"], x["rho"],
                              z_ref=10.0, z_t=zt, z_q=zt, scheme="coare3", n_iter=5,
                              charnock=CHARNOCK_AEROBULK, L_latent=x["L"], return_2m=True,
                              z_diag=zt)
    t, ta = np.asarray(out[5]), np.asarray(x["th_a"])
    assert np.max(np.abs(t - ta) / ta) <= _RTOL


def test_coare3_gradient_finite_calm_stable_column():
    """Exact calm wind over a colder sea (no gustiness: stable) - the bulk
    wind sits on AeroBulk's 0.2 m/s floor; the gradient must stay finite."""
    def loss(u, v, ts):
        r = compute_most_fluxes(u, v, jnp.array([285.0]), jnp.array([6e-3]), ts,
                                jnp.array([7e-3]), jnp.array([1.2]), scheme="coare3")
        return jnp.sum(r[0] + r[2] + r[3] + r[4])
    for g in jax.grad(loss, argnums=(0, 1, 2))(jnp.array([0.0]), jnp.array([0.0]),
                                                jnp.array([280.0])):
        assert np.all(np.isfinite(np.asarray(g)))


def _default_gaps(d):
    """Signed per-state gap (ours at the public defaults - oracle), per height and
    output: Charnock 0.011 and the default neutral width."""
    out = {}
    for h, zt in _HEIGHTS.items():
        ours, ref = _ours(_inputs(d, h), zt, charnock=0.011), _oracle(d, h)
        for k, b in ref.items():
            out[f"{h}_{k}"] = np.asarray(ours[k]) - b
    return out


def test_default_width_is_the_recorded_one():
    import inspect
    assert COARE_NEUTRAL_WIDTH == 1.0e-3
    sig = inspect.signature(compute_most_fluxes).parameters
    assert sig["coare_neutral_width"].default == COARE_NEUTRAL_WIDTH
    assert sig["charnock"].default == 0.011


def test_default_gap_per_state():
    """RULE 5 KNOWN DIFFERENCE (items 44 + 47): at Charnock 0.011 and the default
    neutral width every state's signed gap to AeroBulk equals the stored one."""
    stored = np.load(_GAP_FILE)
    gaps = _default_gaps(_fixture())
    assert sorted(gaps) == sorted(k for k in stored.files if k != "width")
    for k, g in gaps.items():
        np.testing.assert_allclose(g, stored[k], rtol=1e-6, atol=1e-12, err_msg=k)
    assert max(np.abs(stored[k]).max() for k in gaps) > 1e-6


def test_check_grads_through_the_neutral_blend_at_default():
    """Gradient through the neutral blend at the default: near-neutral states
    (|zeta| inside the blend band; eps = 1e-5 K moves zeta by ~1e-6).  Exact
    neutral is excluded: there finite differences cross the still-hard dt sign
    floor and first-guess Richardson sign switch (follow-up).  NOTE: these
    states pass at width 0 too, so this is coverage, not the discriminating
    RULE 5 gradient test (open question: the blend's derivative kink at
    zeta = 0)."""
    n = 4
    ta = jnp.full(n, 290.0)
    q = jnp.full(n, 0.010)

    def f(ts):
        r = compute_most_fluxes(jnp.full(n, 8.0), jnp.zeros(n), ta, q, ts, q,
                                jnp.full(n, 1.2), z_ref=10.0, scheme="coare3")
        return jnp.sum(r[0] + r[2] + r[3] + r[4])

    ts0 = ta + jnp.asarray([-0.02, -0.005, 0.005, 0.02])
    check_grads(f, (ts0,), order=2, modes=("fwd", "rev"), eps=1e-5)
    assert np.all(np.asarray(jax.grad(f)(ts0)) != 0.0)


def test_exchange_cap_below_aerobulk_floor_raises():
    x = [jnp.asarray([v]) for v in (5.0, 0.0, 290.0, 0.01, 291.0, 0.012, 1.2)]
    for scheme in ("coare3", "large_yeager"):
        with pytest.raises(ValueError, match="Cx_min"):
            compute_most_fluxes(*x, scheme=scheme, max_exchange_coeff=0.99e-4)
        compute_most_fluxes(*x, scheme=scheme, max_exchange_coeff=1.0e-4)


if __name__ == "__main__":
    np.savez_compressed(_GAP_FILE, width=COARE_NEUTRAL_WIDTH, **_default_gaps(_fixture()))
    print("wrote", _GAP_FILE)
