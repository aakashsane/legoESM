"""``spectral_rollout(adjoint_steps=k)``: same forecast, gradient from the last k steps only."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.training.neural_gcm_spectral import (
    _scan_adjoint_window, carry_to_spectral_state, spectral_rollout,
    spectral_state_to_carry,
)
from tests.unit.test_spectral_carry_tracer_set import _carry
from tests.unit.test_spectral_phys_state_thread import _rad_zero, _stub_physics


def _toy(a, k, n=7):
    # x <- a*x + idx: the step index must continue across the cut.
    step = lambda x, i: (a * x + i, None)
    return _scan_adjoint_window(step, jnp.asarray(1.0), jnp.arange(n, dtype=float),
                                None, k)[0]


def _toy_ref(a, k, n=7):
    x = jnp.asarray(1.0)
    for i in range(n):
        if i == n - k:
            x = jax.lax.stop_gradient(x)
        x = a * x + i
    return x


@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_scan_window_forward_and_gradient(k):
    assert float(_toy(1.1, k)) == float(_toy(1.1, 0))
    g, g_ref = jax.grad(_toy)(1.1, k), jax.grad(_toy_ref)(1.1, k)
    np.testing.assert_allclose(float(g), float(g_ref), rtol=1e-14)


def test_scan_window_cut_changes_gradient():
    # Non-vacuity: a 1-step window is NOT the full gradient.
    assert abs(float(jax.grad(_toy)(1.1, 1)) - float(jax.grad(_toy)(1.1, 0))) > 1.0


def _setup():
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    state0 = carry_to_spectral_state(_carry(grid.n_lat, grid.n_lon, extras=False), grid)
    return grid, sigma, state0


def _roll(coef, k, grid, sigma, state0, n=6):
    fn = _stub_physics(None, None, heat_per_unit_state=coef)
    final = spectral_rollout(
        state0, fn, grid, sigma, SpectralPEConfig(semi_implicit=True), 600.0, n,
        None, None, rad_physics_fn=_rad_zero, rad_update_interval=3,
        adjoint_steps=k)
    return jnp.mean(jnp.asarray(spectral_state_to_carry(final, grid, sigma).T) ** 2)


def test_stateful_rollout_same_forecast_and_suffix_gradient():
    grid, sigma, state0 = _setup()
    full = _roll(1e-5, 0, grid, sigma, state0)
    assert float(_roll(1e-5, 2, grid, sigma, state0)) == float(full)

    # Reference: leg 1 stopped, physics carry handed on to leg 2.
    def chained(coef):
        fn = _stub_physics(None, None, heat_per_unit_state=coef)
        kw = dict(rad_physics_fn=_rad_zero, rad_update_interval=3)
        cfg = SpectralPEConfig(semi_implicit=True)
        mid, ps = spectral_rollout(state0, fn, grid, sigma, cfg, 600.0, 4,
                                   None, None, return_phys_state=True, **kw)
        mid, ps = jax.lax.stop_gradient((mid, ps))
        out = spectral_rollout(mid, fn, grid, sigma, cfg, 600.0, 2, None, None,
                               phys_state_in=ps, **kw)
        return jnp.mean(jnp.asarray(spectral_state_to_carry(out, grid, sigma).T) ** 2)

    g2 = float(jax.grad(_roll)(1e-5, 2, grid, sigma, state0))
    g0 = float(jax.grad(_roll)(1e-5, 0, grid, sigma, state0))
    np.testing.assert_allclose(g2, float(jax.grad(chained)(1e-5)), rtol=1e-10)
    np.testing.assert_allclose(float(jax.grad(_roll)(1e-5, 6, grid, sigma, state0)),
                               g0, rtol=1e-12)
    assert abs(g2 - g0) > 1e-3 * abs(g0)


def test_adjoint_steps_out_of_range_raises():
    grid, sigma, state0 = _setup()
    with pytest.raises(ValueError, match="adjoint_steps"):
        _roll(1e-5, 7, grid, sigma, state0)
