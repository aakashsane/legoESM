"""Mass-weighted flux form of the MPAS vertical del4 tracer filter.

The projected form removes the column's index-space non-conservation as a
UNIFORM mixing-ratio offset at every level.  For water vapour, four decades
drier in the stratosphere than at the surface, that offset is a large relative
sink aloft (e12_year, 2026-10-07: -0.4 ppmv/day at 45-70N, -3 ppmv/day in the
tropics, stratosphere dried to ~1 ppmv).  The flux form conserves column by
column through adjacent-layer fluxes only.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    vertical_del4_T_tendency, vertical_del4_tracer_tendency_flux)

NU = 2.0e-6
DAY = 86400.0


@pytest.fixture(autouse=True)
def fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _stretched_column():
    """32 layers: thin at the top (~2 hPa), thick near the surface (~60 hPa),
    water vapour from 15 g/kg at the surface to a flat 4 ppmv (2.5e-6 kg/kg)
    stratosphere above level 14 -- the shape that exposed the drain."""
    dp = jnp.asarray(np.geomspace(200.0, 6000.0, 32))          # Pa, top first
    k = np.arange(32)
    q = np.where(k < 14, 2.5e-6, 2.5e-6 * np.exp((k - 13) * np.log(1.5e-2 / 2.5e-6) / 18.0))
    return jnp.asarray(q), dp


def test_flux_form_conserves_each_column_exactly():
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.uniform(1e-7, 1e-2, size=(7, 32)))
    dp = jnp.asarray(rng.uniform(100.0, 6000.0, size=(7, 32)))
    t = vertical_del4_tracer_tendency_flux(q, NU, dp)
    col = np.asarray((t * dp).sum(-1))
    scale = np.asarray((jnp.abs(t) * dp).sum(-1))
    assert np.all(np.abs(col) <= 1e-13 * scale)
    assert np.all(scale > 0)                                   # control: it acts


def test_flux_form_does_not_drain_the_dry_stratosphere():
    """The regression: on a stretched column the projected form adds a uniform
    REMOTE offset (sign set by the tropospheric profile and the layer masses;
    in e12_year it was negative, a sink) that changes the flat stratosphere's
    vapour by >5 %/day; the flux form leaves the flat region untouched (its
    fourth difference is zero there and no remote correction is added)."""
    q, dp = _stretched_column()
    proj = np.asarray(vertical_del4_T_tendency(q, NU, dp))
    flux = np.asarray(vertical_del4_tracer_tendency_flux(q, NU, dp))
    flat = slice(2, 11)                                        # well inside the flat layer
    rel_proj = proj[flat] / np.asarray(q)[flat] * DAY
    rel_flux = flux[flat] / np.asarray(q)[flat] * DAY
    assert np.all(np.abs(rel_proj) > 0.05)                     # control: the defect exists
    assert np.max(np.abs(rel_flux)) < 1e-12
    # both conserve the column
    assert abs(float((proj * dp).sum())) <= 1e-12 * float((np.abs(proj) * dp).sum())
    assert abs(float((flux * dp).sum())) <= 1e-12 * float((np.abs(flux) * dp).sum())


def test_uniform_grid_matches_projected_form():
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.uniform(1e-6, 1e-2, size=(5, 20)))
    dp = jnp.full((20,), 5000.0)
    np.testing.assert_allclose(np.asarray(vertical_del4_tracer_tendency_flux(q, NU, dp)),
                               np.asarray(vertical_del4_T_tendency(q, NU, dp)),
                               rtol=1e-12, atol=1e-24)


def test_linear_profile_and_checkerboard():
    dp = jnp.asarray(np.geomspace(300.0, 5000.0, 16))
    lin = jnp.linspace(1e-6, 1e-3, 16)
    assert float(jnp.max(jnp.abs(vertical_del4_tracer_tendency_flux(lin, NU, dp)))) < 1e-22
    cb = 1e-3 + 1e-4 * (-1.0) ** jnp.arange(16)
    t = vertical_del4_tracer_tendency_flux(cb, NU, dp)
    # damps the 2-grid mode: tendency opposes the anomaly in the interior
    assert bool(jnp.all(t[2:-2] * (cb[2:-2] - 1e-3) < 0))


def test_nonpositive_layer_mass_gives_zero_not_inf():
    q = jnp.asarray(np.linspace(1e-5, 1e-2, 8))
    dp = jnp.asarray([500.0, 800.0, 1200.0, 0.0, 1500.0, 2000.0, 2500.0, 3000.0])
    t = vertical_del4_tracer_tendency_flux(q ** 2, NU, dp)
    assert bool(jnp.all(jnp.isfinite(t))) and float(t[3]) == 0.0


def _tracer_state():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel)
    from legoesm.atmosphere.idealized.held_suarez_topo import held_suarez_topo_init_mpas
    from legoesm.core.field import Field
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(3)
    coord = create_sigma_coordinate(10)
    cfg = MPASPrimitiveEquationConfig(fix_mass=False, nu_vert4_T=NU)
    state = held_suarez_topo_init_mpas(mesh, coord)
    model = MPASPrimitiveEquationModel(mesh, coord, cfg)
    for _ in range(2):
        state = model.step(state, 300.0)
    n = state.T.data.shape
    k = jnp.arange(n[1])
    qv = jnp.broadcast_to(1e-2 * jnp.exp(-(n[1] - 1 - k) * 1.2), n)   # dry aloft
    qc = jnp.broadcast_to(1e-5 * (1.0 + 0.5 * (-1.0) ** k), n)
    mk = lambda a, nm: Field(data=a, name=nm, dims=("nCells", "nlev"), units="kg/kg")  # noqa: E731
    return mesh, coord, cfg, state._replace(tracers={"q_v": mk(qv, "q_v"), "q_c": mk(qc, "q_c")})


def test_dycore_selects_flux_form_per_tracer():
    """The dycore must route ONLY the named tracers through the flux form."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import mpas_hydrostatic_tendencies
    mesh, coord, cfg, state = _tracer_state()
    base = mpas_hydrostatic_tendencies(state, mesh, coord, cfg, None, 0.0)
    sel = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg._replace(vert4_flux_form_tracers=("q_v",)), None, 0.0)
    qv = state.tracers["q_v"].data
    expect = (vertical_del4_tracer_tendency_flux(qv, NU, coord.dsigma)
              - vertical_del4_T_tendency(qv, NU, coord.dsigma))
    assert float(jnp.max(jnp.abs(expect))) > 0.0                # control: forms differ here
    np.testing.assert_allclose(
        np.asarray(sel.tracer_tendencies["q_v"].data - base.tracer_tendencies["q_v"].data),
        np.asarray(expect), rtol=1e-9, atol=1e-22)
    np.testing.assert_array_equal(np.asarray(sel.tracer_tendencies["q_c"].data),
                                  np.asarray(base.tracer_tendencies["q_c"].data))


def test_unknown_tracer_name_raises():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import mpas_hydrostatic_tendencies
    mesh, coord, cfg, state = _tracer_state()
    with pytest.raises(ValueError, match="not tracers"):
        mpas_hydrostatic_tendencies(
            state, mesh, coord, cfg._replace(vert4_flux_form_tracers=("q_x",)), None, 0.0)
