"""Direct tiny-grid controls for NEMO mode-32 momentum diffusion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_vertex_mask,
    nemo_ldf_lap_viscosity_e3_cgrid,
    nemo_smagorinsky_bilaplacian_coefficients_cgrid,
    nemo_smagorinsky_bilaplacian_tendency_e3_cgrid,
)


def _case():
    ny, nx, nz = 5, 7, 2
    grid = create_beta_plane_cgrid_geometry(
        ny, nx, dx_m=25_000.0, dy_m=25_000.0, f0=1.0e-4, beta=2.0e-11
    )
    mask = jnp.ones((ny, nx), dtype=jnp.float64)
    mask = mask.at[[0, -1], :].set(0.0).at[:, [0, -1]].set(0.0)
    u_mask = jnp.concatenate(
        [jnp.zeros((ny, 1)), mask[:, :-1] * mask[:, 1:], jnp.zeros((ny, 1))], axis=1
    )
    v_mask = jnp.concatenate([jnp.zeros((1, nx)), mask[:-1] * mask[1:], jnp.zeros((1, nx))], axis=0)
    cell3 = jnp.broadcast_to(mask[..., None], (ny, nx, nz))
    u3 = jnp.broadcast_to(u_mask[..., None], (ny, nx + 1, nz))
    v3 = jnp.broadcast_to(v_mask[..., None], (ny + 1, nx, nz))
    vertex3 = jnp.broadcast_to(
        compute_vertex_mask(mask, grid=grid)[..., None], (ny + 1, nx + 1, nz)
    )
    return grid, mask, u_mask[..., None], v_mask[..., None], cell3, vertex3, u3, v3


def test_mode32_zero_flow_is_exact_zero_with_finite_gradient():
    grid, mask, um, vm, cell3, fm, u, v = _case()
    u = jnp.zeros_like(u)
    v = jnp.zeros_like(v)

    def total(uu, vv):
        ahmt, ahmf = nemo_smagorinsky_bilaplacian_coefficients_cgrid(
            uu,
            vv,
            grid,
            3.5,
            3600.0,
            mask=mask,
            cell_mask=cell3,
            u_mask=um,
            v_mask=vm,
            vertex_mask=fm,
        )
        return jnp.sum(ahmt) + jnp.sum(ahmf)

    assert float(total(u, v)) == 0.0
    gu, gv = jax.grad(total, argnums=(0, 1))(u, v)
    assert np.all(np.isfinite(np.asarray(gu)))
    assert np.all(np.isfinite(np.asarray(gv)))

    @jax.jit
    def traced_dynamic_dt(dt_s):
        return nemo_smagorinsky_bilaplacian_coefficients_cgrid(
            u,
            v,
            grid,
            3.5,
            dt_s,
            mask=mask,
            cell_mask=cell3,
            u_mask=um,
            v_mask=vm,
            vertex_mask=fm,
        )

    traced_t, traced_f = traced_dynamic_dt(jnp.asarray(3600.0))
    assert float(jnp.sum(traced_t) + jnp.sum(traced_f)) == 0.0


def test_mode32_bilaplacian_is_two_e3_laplacians_with_negative_second_pass():
    grid, mask, um, vm, cell3, fm, u_masked, v_masked = _case()
    jj_u, ii_u, kk_u = jnp.indices(u_masked.shape)
    jj_v, ii_v, kk_v = jnp.indices(v_masked.shape)
    u = (0.04 * jnp.sin(0.31 * ii_u + 0.17 * jj_u + kk_u)) * u_masked
    v = (0.03 * jnp.cos(0.27 * ii_v - 0.13 * jj_v + kk_v)) * v_masked
    ahmt, ahmf = nemo_smagorinsky_bilaplacian_coefficients_cgrid(
        u, v, grid, 3.5, 3600.0, mask=mask, cell_mask=cell3, u_mask=um, v_mask=vm, vertex_mask=fm
    )
    h = jnp.full(cell3.shape, 100.0) * cell3
    got_u, got_v = nemo_smagorinsky_bilaplacian_tendency_e3_cgrid(
        u, v, grid, ahmt, ahmf, h, mask=mask, u_mask=um, v_mask=vm, vertex_mask=fm
    )
    lap_u, lap_v = nemo_ldf_lap_viscosity_e3_cgrid(
        u, v, grid, ahmt, ahmf, h, mask=mask, u_mask=um, v_mask=vm, vertex_mask=fm
    )
    second_u, second_v = nemo_ldf_lap_viscosity_e3_cgrid(
        lap_u, lap_v, grid, ahmt, ahmf, h, mask=mask, u_mask=um, v_mask=vm, vertex_mask=fm
    )
    np.testing.assert_array_equal(np.asarray(got_u), -np.asarray(second_u))
    np.testing.assert_array_equal(np.asarray(got_v), -np.asarray(second_v))
    assert np.max(np.abs(np.asarray(got_u))) > 0.0
    assert np.max(np.abs(np.asarray(got_v))) > 0.0


@pytest.mark.slow
def test_mode32_card_executes_one_complete_jitted_gyre_step():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_gyre_zco_card,
        gyre_surface_forcings,
    )

    namelist = GYRE_R4_SMAG_NAMELIST._replace(
        nn_GYRE=1,
        rn_Dt_s=14_400.0,
        rn_Ld_m=100_000.0,
        nn_itend=1,
        nn_stock=1,
        nn_write=1,
    )
    card = build_gyre_zco_card(namelist)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    freshwater, surface = gyre_surface_forcings(card, card.recipe.initial_state, 1)
    state = model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater, surface_forcing=surface
    )
    jax.block_until_ready(state.eta.data)
    for name in ("T", "S", "u", "v", "eta"):
        assert np.all(np.isfinite(np.asarray(getattr(state, name).data)))
