"""CAM-style dycore total-energy fixer (review 2026-10-10 F15).

The KE the dycore damping removes (MPAS del2 sponge / del4 / div4, the duo's
d_con=0 damping) and any numerical dissipation used to leave the total energy
for good.  ``DycoreConfig.energy_fixer`` returns the dynamics window's energy
change as a uniform heating (``core.conservation.energy_fixer_dT``), the
analogue of CAM's ``check_energy_fix``.

Pinned here (x64): the global energy ``sum((c_pd*T + KE) dp/g + phis*p_s/g)``
is unchanged by a step with the fixer on (to round-off) and is NOT unchanged
with it off (non-vacuity), on the Voronoi (MPAS) lane and the cubed-sphere
(FV3-duo column) lane; the fixer is differentiable; and every lane that does
not run it refuses ``energy_fixer=True``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.conservation import (  # noqa: E402
    dycore_column_energy,
    energy_fixer_dT,
)
from legoesm.core.operators_voronoi import kinetic_energy_cell_3d  # noqa: E402


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


def test_dT_restores_the_energy_exactly():
    """E(after + dT) == E(before) for arbitrary columns (dp independent of T)."""
    rng = np.random.default_rng(1)
    ncol, nlev = 40, 12
    area = jnp.asarray(rng.random(ncol) + 0.5)
    dp = jnp.asarray(rng.random((ncol, nlev)) * 4000.0 + 500.0)
    p_s = jnp.sum(dp, axis=-1) + 200.0
    phis = jnp.asarray(rng.random(ncol) * 2e4)

    def E(T, ke):
        return float(jnp.sum(dycore_column_energy(T, ke, dp, phis, p_s) * area))

    T0 = jnp.asarray(250.0 + 20.0 * rng.random((ncol, nlev)))
    ke0 = jnp.asarray(200.0 * rng.random((ncol, nlev)))
    T1, ke1 = T0 - 0.01, ke0 * 0.97            # a lossy "dynamics window"
    mass = float(jnp.sum(jnp.sum(dp, -1) / constants.g * area))
    dT = energy_fixer_dT(E(T0, ke0), E(T1, ke1), mass)
    assert float(dT) > 0.0                     # a loss is returned as warming
    np.testing.assert_allclose(E(T1 + dT, ke1), E(T0, ke0), rtol=1e-14)


# ---------------------------------------------------------------------------
# MPAS (Voronoi) lane
# ---------------------------------------------------------------------------

NLEV, DT = 8, 600.0


@pytest.fixture(scope="module")
def mpas():
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
    mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
    sigma = create_sigma_coordinate(NLEV)
    return mesh, sigma, baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)


def _mpas_model(mesh, sigma, fixer):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel)
    # damping ON (the sink the fixer closes); no mass fixer, so the step's
    # only energy change is the dynamics window's
    cfg = MPASPrimitiveEquationConfig(
        nu_del2=2.0e5, nu_del4=1.0e19, nu_div4=1.0e19,
        sponge_del2_top_layers=2, sponge_del2_top_factor=8.0,
        fix_mass=False, energy_fixer=fixer)
    return MPASPrimitiveEquationModel(mesh, sigma, cfg)


def _mpas_energy(state, mesh, sigma):
    ph = sigma.pressure_at_half(state.p_s.data)
    e = dycore_column_energy(state.T.data,
                             kinetic_energy_cell_3d(state.u.data, mesh),
                             ph[..., 1:] - ph[..., :-1], state.phis.data,
                             state.p_s.data)
    return float(jnp.sum(e * mesh.areaCell))


def test_mpas_step_conserves_total_energy_with_the_fixer(mpas):
    mesh, sigma, s0 = mpas
    e0 = _mpas_energy(s0, mesh, sigma)
    off = _mpas_model(mesh, sigma, False).step(s0, DT)
    on = _mpas_model(mesh, sigma, True).step(s0, DT)
    rel_off = (_mpas_energy(off, mesh, sigma) - e0) / e0
    rel_on = (_mpas_energy(on, mesh, sigma) - e0) / e0
    # non-vacuous: the damped dynamics LOSES energy without the fixer...
    assert rel_off < -1e-10, rel_off
    # ...and the fixer returns it to round-off
    assert abs(rel_on) < 1e-13, rel_on
    # the fix is a uniform warming, nothing else moves
    dT = np.asarray(on.T.data - off.T.data)
    assert np.all(dT > 0.0) and np.ptp(dT) < 1e-9 * dT.max()
    np.testing.assert_array_equal(np.asarray(on.u.data), np.asarray(off.u.data))


def test_mpas_fixer_is_differentiable(mpas):
    mesh, sigma, s0 = mpas
    model = _mpas_model(mesh, sigma, True)

    def loss(T):
        out = model._step_jit(s0._replace(T=s0.T.replace(data=T)), DT,
                              None, None, None, None)[0]
        return jnp.sum(out.T.data * mesh.areaCell[:, None])

    g = jax.grad(loss)(s0.T.data)
    assert np.all(np.isfinite(np.asarray(g)))


def test_voronoi_helper_matches_serial_weights_under_a_split_area(mpas):
    """The MPI form (owned-area weight + a cross-rank sum of the three local
    integrals) is the serial result: two disjoint halves summed == whole."""
    from legoesm.core.conservation import voronoi_energy_fixer_dT
    mesh, sigma, s0 = mpas
    s1 = _mpas_model(mesh, sigma, False).step(s0, DT)
    whole = float(voronoi_energy_fixer_dT(s0, s1, mesh, sigma, mesh.areaCell))
    half = jnp.arange(mesh.nCells) < mesh.nCells // 2
    parts = []

    def capture(x):          # stands in for the allreduce: record the partials
        parts.append(x)
        return x
    for m in (half, ~half):
        voronoi_energy_fixer_dT(s0, s1, mesh, sigma,
                                jnp.where(m, mesh.areaCell, 0.0), sum_fn=capture)
    tot = parts[0] + parts[1]
    split = float(energy_fixer_dT(tot[0], tot[1], tot[2]))
    np.testing.assert_allclose(split, whole, rtol=1e-12)
    assert whole > 0.0


# ---------------------------------------------------------------------------
# FV3-duo column lane (cubed sphere, structured)
# ---------------------------------------------------------------------------

DUO_N, DUO_NG, DUO_KM, DUO_DT = 12, 3, 5, 120.0


@pytest.fixture(scope="module")
def duo():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    grid = create_fv3_duo_grid(DUO_N, DUO_NG)
    dyn = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=DUO_KM, n_split=2))
    return dyn, dyn.dcmip16_initial_state(n_tracers=3)


def _duo_energy(col, st):
    ph = col.sigma_coord.pressure_at_half(st.p_s.data)
    ke = 0.5 * (st.u.data ** 2 + st.v.data ** 2)
    e = dycore_column_energy(st.T.data, ke, ph[..., 1:] - ph[..., :-1],
                             st.phis.data, st.p_s.data)
    return float(jnp.sum(e * jnp.asarray(col.mesh.areaCell)))


def _zero_physics(state, mesh, coord, phys_state=None, forcing=None):
    from legoesm.core.state import HydrostaticTendencies
    z2 = jnp.zeros_like(state.p_s.data)
    f = lambda like, nm: like.replace(data=jnp.zeros_like(like.data), name=nm)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=f(state.u, "du"), dv_dt=f(state.v, "dv"), dT_dt=f(state.T, "dT"),
        dp_s_dt=state.p_s.replace(data=z2, name="dps"),
        dphis_dt=state.phis.replace(data=z2, name="dphis"))


@pytest.mark.parametrize("physics", [None, _zero_physics])
def test_duo_column_step_conserves_total_energy_with_the_fixer(duo, physics):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnModel
    dyn, ic = duo
    rel = {}
    for fixer in (False, True):
        col = FV3DuoColumnModel(dyn, energy_fixer=fixer)
        s0 = col.from_bundle(ic)
        e0 = _duo_energy(col, s0)
        s1 = col.step(s0, DUO_DT, physics_fn=physics)
        rel[fixer] = (_duo_energy(col, s1) - e0) / e0
    assert abs(rel[False]) > 1e-10, rel        # non-vacuous
    assert abs(rel[True]) < 1e-13, rel


# ---------------------------------------------------------------------------
# lanes that do not run it refuse it
# ---------------------------------------------------------------------------

def test_unwired_lanes_refuse_the_fixer():
    from legoesm.driver.component_factory import (
        energy_fixer_wired, refuse_unwired_energy_fixer)
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    assert DycoreConfig().energy_fixer is True          # RULE 3: default = fix
    on = ExperimentConfig()
    assert energy_fixer_wired(on, "mpas_primitive_equations")
    duo_col = on._replace(dycore=on.dycore._replace(
        discretization="fv3_duo", fv3_duo_column_lane=True))
    assert energy_fixer_wired(duo_col, "fv3_duo_primitive_equations")
    for solver in ("cdgrid_primitive_equations",
                   "latlon_cgrid_primitive_equations",
                   "spectral_primitive_equations",
                   "mpas_compressible_euler", "fv3_duo_primitive_equations"):
        with pytest.raises(ValueError, match="energy_fixer"):
            refuse_unwired_energy_fixer(on, solver)
        off = on._replace(dycore=on.dycore._replace(energy_fixer=False))
        refuse_unwired_energy_fixer(off, solver)        # explicit opt-out
