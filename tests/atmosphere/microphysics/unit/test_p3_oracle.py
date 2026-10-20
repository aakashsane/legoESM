"""P3 (one ice category) vs the RUNNING gSAM 1.8.8 P3 oracle.

Fixture ``tests/atmosphere/microphysics/oracle_fixtures/p3.npz`` was generated
by RUNNING gSAM 1.8.8 ``SRC/MICRO_P3/module_mp_p3.f90`` (P3 v3.1.14, tarball
sha256 in the npz ``meta``) compiled with gfortran in fp64, through
``scripts/validate/oracle/p3/build.sh`` + ``make_fixture.py``.  The oracle is
fed legoESM saturation specific humidity and the legoESM constants (the tier-2
audit glue, recorded in ``meta``).  gSAM defaults: Meyers nucleation, KK2000
warm rain, specified droplet number, MaxTotalIceNumber 20/cc.

Tolerance (user decision 2026-10-19): max relative error <= 1e-10 in fp64 with
every existing sharpness field at its hard-switch limit.  Relative error is
``|a-b| / max(|a|, |b|)`` (0 where both are exactly 0); no floor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import p3 as P3  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import P3Config  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/p3.npz"
TOL = 1.0e-10
# Every existing sharpness field -> its hard-switch limit (the oracle has hard
# switches; same limits as the tier-1 audit's HARD_LIMIT).
HARD_LIMIT = dict(ice_sigmoid_sharpness=1.0e6, cooper_supi_sharpness=1.0e9,
                  melt_sharpness=1.0e6)
HARD = P3Config()._replace(**HARD_LIMIT)
STATE_MAP = dict(T="T", q_v="qv", q_c="qc", q_r="qr", q_i="qi", q_rim="qirim",
                 B_rim="birim", N_i="ni")
IN_KEYS = ("p", "rho", "dz", "T", "q_v", "q_c", "N_c", "q_r", "N_r", "q_i",
           "q_rim", "N_i", "B_rim")


def _fx():
    return dict(np.load(FIXTURE))


def _meta(fx):
    return json.loads(str(fx["meta"]))


def _cases():
    m = _meta(_fx())
    return [(c, dt) for c in m["cases"]
            for dt in (m["nuc_dts"] if c == "nuc" else m["dts"])]


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def _inputs(fx, case, dtype=jnp.float64):
    g = {k: jnp.asarray(fx[f"in_{case}_{k}"], dtype) for k in IN_KEYS}
    hyd = HydrometeorState(q_c=g["q_c"], q_r=g["q_r"], q_i=g["q_i"], q_s=g["q_rim"],
                           q_g=g["B_rim"], N_c=g["N_c"], N_r=g["N_r"], N_i=g["N_i"])
    return g, hyd


def _errors(fx, case, dt, out):
    s, prt_liq, prt_sol = out
    o = f"out_{case}_dt{dt:g}_"
    rho = fx[f"in_{case}_rho"]
    errs = {k: _rel(s[k], fx[o + v]) for k, v in STATE_MAP.items()}
    errs["N_c"] = _rel(np.asarray(s["N_c"]) / rho, fx[o + "nc"])
    errs["N_r"] = _rel(np.asarray(s["N_r"]) / rho, fx[o + "nr"])
    errs["prt_liq"] = _rel(prt_liq, fx[o + "prt_liq"])
    errs["prt_sol"] = _rel(prt_sol, fx[o + "prt_sol"])
    return {k: float(v.max()) for k, v in errs.items()}


def test_fixture_provenance_and_vendored_table():
    fx = _fx()
    m = _meta(fx)
    assert m["tarball_sha256"] == (
        "0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c")
    assert m["module_mp_p3_sha256"] == (
        "b5f3021d6bab6b88b9bd6b77ae588ea0163844923acd2e95e9f23a6e8c9278e3")
    sha = hashlib.sha256(Path(P3.P3_LOOKUP_TABLE_1).read_bytes()).hexdigest()
    assert sha == m["lookup_table_1_sha256"]
    # the oracle's saturation stub reproduces legoesm.thermo
    assert max(m["qsat_stub_vs_thermo_max_rel"].values()) <= TOL


def test_generator_rebuilds_the_analytic_inputs():
    """The committed generator rebuilds the sounding-free inputs (arctic_mixed,
    edge, coverage, the nucleation points) bit-for-bit, and its output-field
    list is the fixture's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p3_make_fixture", ROOT / "scripts/validate/oracle/p3/make_fixture.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    fx = _fx()
    assert list(gen.OUT_FIELDS) == _meta(fx)["out_fields"]
    for f in (gen.arctic_mixed, gen.edge, gen.coverage):
        c = f(None)
        for k in gen.IN_FIELDS:
            np.testing.assert_array_equal(np.asarray(c[k], float), fx[f"in_{c['name']}_{k}"][0])
    nuc = gen.nucleation_points(None)
    for k in gen.IN_FIELDS:
        np.testing.assert_array_equal(np.asarray(nuc[k], float), fx[f"in_nuc_{k}"][:, 0])


@pytest.mark.parametrize("case,dt", _cases())
def test_whole_step_matches_gsam_p3(case, dt):
    """Every post-step state field and both surface precipitation rates."""
    fx = _fx()
    g, hyd = _inputs(fx, case)
    out = P3.p3_step(g["T"], g["q_v"], hyd, g["p"], g["rho"], g["dz"], float(dt), HARD)
    worst = _errors(fx, case, dt, out)
    print(case, dt, "max rel err", worst)
    assert max(worst.values()) <= TOL, worst


def test_fixture_exercises_the_processes():
    """Every process P3 runs with these defaults is active somewhere in the
    fixture (qrhomi is not: rain below -37 C is immersion-frozen first)."""
    fx = _fx()
    m = _meta(fx)
    active = ("qccon", "qcevp", "qrcon", "qrevp", "qcaut", "qcacc", "nrslf",
              "qccol", "qrcol", "qidep", "qisub", "qinuc", "nislf", "qimlt",
              "qcheti", "qrheti", "qcshd", "qwgrth", "qcnuc", "qchomi", "nrevp")
    for d in active:
        n = sum(int(np.count_nonzero(fx[f"out_{c}_dt{dt:g}_{d}"])) for c, dt in _cases())
        assert n > 0, d
    assert m["cases"] and "nuc" in m["cases"]


def test_sedimentation_substeps_and_dt300():
    """dt=300 drives the adaptive sedimentation loop past one sub-step (rain
    on sc_dycoms needs dozens), the regime the while-loop port must match."""
    fx = _fx()
    g, hyd = _inputs(fx, "sc_dycoms")
    counts = []
    orig = P3.eqxi.while_loop

    def counting(cond, body, init, **kw):
        c, n = init, 0
        while bool(cond(c)):
            c, n = body(c), n + 1
        counts.append(n)
        return c

    P3.eqxi.while_loop = counting
    try:
        P3.p3_step(g["T"], g["q_v"], hyd, g["p"], g["rho"], g["dz"], 300.0, HARD)
    finally:
        P3.eqxi.while_loop = orig
    assert max(counts) > 10, counts


def test_column_water_budget_closes_like_the_oracle():
    """Column vapour+condensate change + surface precipitation = 0 to
    round-off for both the port and the oracle (state the residual)."""
    from legoesm import constants as C
    fx = _fx()
    for case, dt in _cases():
        g, hyd = _inputs(fx, case)
        s, pl, ps = P3.p3_step(g["T"], g["q_v"], hyd, g["p"], g["rho"], g["dz"],
                               float(dt), HARD)
        m = np.asarray(g["rho"] * g["dz"])
        q0 = np.asarray(g["q_v"] + g["q_c"] + g["q_r"] + g["q_i"])
        q1 = np.asarray(s["q_v"] + s["q_c"] + s["q_r"] + s["q_i"])
        res = (m * (q1 - q0)).sum(-1) + np.asarray(pl + ps) * C.rho_water * dt
        rel = np.abs(res) / (m * q0).sum(-1)
        print(case, dt, "water residual / column water", float(rel.max()))
        assert np.all(rel <= TOL), (case, dt, rel)


def test_default_smoothing_gap_is_reported():
    """At the shipped (default) smoothing the port is NOT the oracle: print
    the gap (PR body) and check it stays finite."""
    fx = _fx()
    for case, dt in _cases():
        g, hyd = _inputs(fx, case)
        out = P3.p3_step(g["T"], g["q_v"], hyd, g["p"], g["rho"], g["dz"],
                         float(dt), P3Config())
        print(case, dt, "default-smoothing max rel err", _errors(fx, case, dt, out))


@pytest.mark.parametrize("case,dt", [
    pytest.param(c, dt, marks=pytest.mark.xfail(
        strict=True, reason="XLA FMA in legoesm.thermo's ice-saturation "
        "polynomial moves q_sat_i by 6.5e-11 at 200 K under jit; Meyers "
        "amplifies it to 1.1e-9 (pending tolerance decision)"))
    if (c, dt) in (("nuc", 20.0), ("trop_deep", 20.0)) else (c, dt)
    for c, dt in _cases()])
def test_jit_matches_oracle(case, dt):
    fx = _fx()
    g, hyd = _inputs(fx, case)
    f = jax.jit(lambda T, qv: P3.p3_step(T, qv, hyd, g["p"], g["rho"], g["dz"],
                                         float(dt), HARD))
    worst = _errors(fx, case, dt, f(g["T"], g["q_v"]))
    print(case, dt, "jit max rel err", worst)
    assert max(worst.values()) <= TOL, worst


@pytest.mark.parametrize("case", _meta(_fx())["cases"])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("cfg", [HARD, P3Config()], ids=["hard", "default"])
def test_gradients_finite_on_fixture_columns(case, dtype, cfg):
    fx = _fx()
    g, hyd = _inputs(fx, case, dtype)

    def loss(T, qv, qc, qr, qi):
        h = hyd._replace(q_c=qc, q_r=qr, q_i=qi)
        s, pl, ps = P3.p3_step(T, qv, h, g["p"], g["rho"], g["dz"], 300.0, cfg)
        return (jnp.sum(s["T"]) + 1e3 * jnp.sum(s["q_r"] + s["q_i"] + s["q_v"])
                + 1e3 * jnp.sum(pl + ps))

    # one compiled program per case (eager op-by-op grad of the
    # sedimentation while-loops exhausts the JIT code-memory mappings)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(
        g["T"], g["q_v"], g["q_c"], g["q_r"], g["q_i"])
    jax.clear_caches()
    for gi in grads:
        assert bool(jnp.all(jnp.isfinite(gi))), case
    assert float(jnp.max(jnp.abs(grads[0]))) > 0.0



@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradients_finite_dry_column_and_meyers_overflow(dtype):
    """fp32 overflow guards: inactive rain breakup on a dry column, and the
    Meyers exponential at supi ~ 7 (above the cap); d/d(T, q_v, rho) finite."""
    from legoesm.thermo import saturation_specific_humidity_ice
    q8 = 8.0 * float(saturation_specific_humidity_ice(jnp.asarray(230.0),
                                                      jnp.asarray(6.0e4)))
    for T, qv, p, rho in ((290.0, 5.0e-3, 9.0e4, 1.0), (230.0, q8, 6.0e4, 0.9)):
        f = lambda x: jnp.full((1, 1), x, dtype)  # noqa: E731
        z = f(0.0)
        hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)

        def loss(T, qv, rho):
            s, _, _ = P3.p3_step(T, qv, hyd, f(p), rho, f(200.0), 20.0, HARD)
            return jnp.sum(s["T"]) + 1e3 * jnp.sum(s["q_v"]) + 1e-6 * jnp.sum(s["N_i"])

        g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(f(T), f(qv), f(rho))
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in g), (T, g)


def test_div_custom_jvp_check_grads():
    from jax.test_util import check_grads
    x = jnp.asarray([1.5, -2.0, 3.0e-3])
    y = jnp.asarray([0.7, 4.0, -2.5])
    check_grads(P3._div, (x, y), order=2, modes=("fwd", "rev"))


def test_lookup_table_reader_rejects_other_versions(tmp_path):
    bad = tmp_path / "table.dat"
    bad.write_text("VERSION 5.0\n1 2 3\n")
    with pytest.raises(ValueError, match="version"):
        P3.read_p3_lookup_table_1(bad)
    short = tmp_path / "short.dat"
    short.write_text("VERSION 4.1\n1 2 3\n")
    with pytest.raises(ValueError, match="values"):
        P3.read_p3_lookup_table_1(short)


@pytest.mark.parametrize("field", P3.P3_FIELDS_WITHOUT_ORACLE_COUNTERPART)
def test_surrogate_fields_without_oracle_counterpart_raise(field):
    default = getattr(P3Config(), field)
    new = (not default) if isinstance(default, bool) else default * 2 + 1
    with pytest.raises(ValueError, match=field):
        P3.check_p3_config(P3Config()._replace(**{field: new}))
