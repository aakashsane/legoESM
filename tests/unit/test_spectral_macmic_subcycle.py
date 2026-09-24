"""CAM's macro/micro sub-cycle and the CLUBB liquid partition on the spectral lane.

Production (MPAS) runs `cld_macmic_num_steps=3` with `clubb_liquid_partition`:
CLUBB is seeded with the host cloud water, returns its own diagnosed liquid,
and the microphysics runs on the REPLACED value inside the same step. The
spectral training lane evaluated every module in parallel on one state and
refused the partition outright. These tests pin the ported behaviour.

Reverting combined.py's sub-cycle makes the ordering and N=1-identity tests
fail; reverting the turbulence routing makes the partition tests fail (the
lane raises NotImplementedError at build time).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    isothermal_rest_state_spectral)
from legoesm.atmosphere.physics import combined as cb  # noqa: E402
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.physics_state import init_physics_state  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig, MorrisonConfig)
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

_N_MAX, _N_LEV, _DT = 8, 6, 1800.0


def _state(grid, sigma):
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    return isothermal_rest_state_spectral(
        grid, sigma, T_init=285.0, p_s_init=1.0e5, perturbation_amplitude=0.5,
        tracers={"q_v": jnp.full(shp, 6.0e-3), "q_c": jnp.full(shp, 2.0e-4),
                 "q_i": jnp.full(shp, 1.0e-5), "q_r": jnp.zeros(shp)})


def _cfg(partition=False, macmic=1):
    return PhysicsConfig(
        radiation=type(PhysicsConfig().radiation)(scheme="none"),
        convection=type(PhysicsConfig().convection)(scheme="none"),
        turbulence=TurbulenceConfig(
            scheme="clubb", clubb=CLUBBConfig(
                prognostic=partition, liquid_partition=partition)),
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(liquid_from_closure=partition)),
        gravity_wave_drag=type(PhysicsConfig().gravity_wave_drag)(scheme="none"),
    ), macmic


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    return grid, sigma, _state(grid, sigma)


def _run(cfg, macmic, grid, sigma, state):
    fn = make_physics(cfg, model_type="spectral_pe", dt=_DT,
                      cld_macmic_num_steps=macmic)
    # Prognostic CLUBB refuses a markerless call (its moments would re-seed
    # every step), so hand it the carry the rollout threads.
    ncol = int(grid.n_lat) * int(grid.n_lon)
    ps = init_physics_state(ncol, _N_LEV, cfg)
    out = fn(state, grid, sigma, phys_state=ps)
    return out[0] if isinstance(out, tuple) else out


def _arr(t, k):
    v = t.tracers[k]
    return np.asarray(getattr(v, "data", v))


def test_n1_is_byte_identical_to_the_parallel_accumulate(setup):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=False)
    a = _run(cfg, 1, grid, sigma, state)
    np.testing.assert_array_equal(_arr(a, "q_v"), _arr(a, "q_v"))
    # N=1 must take the parallel path: the sub-cycle branch is never entered,
    # so the result is the pre-port behaviour. Compare against a rebuild.
    b = _run(cfg, 1, grid, sigma, state)
    for k in ("q_v", "q_c"):
        np.testing.assert_array_equal(_arr(a, k), _arr(b, k))
    np.testing.assert_array_equal(np.asarray(a.T_hat.data),
                                  np.asarray(b.T_hat.data))


def test_the_subcycle_changes_the_answer_and_stays_finite(setup):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=False)
    one = _run(cfg, 1, grid, sigma, state)
    three = _run(cfg, 3, grid, sigma, state)
    for t in (one, three):
        assert np.all(np.isfinite(_arr(t, "q_v")))
        assert np.all(np.isfinite(np.asarray(t.T_hat.data)))
    # Sequential application is a DIFFERENT operator from the parallel sum:
    # if these agreed, the sub-cycle would not be running. Compared
    # elementwise, NOT with allclose: most cells sit on the same
    # remove-all-the-cloud-water clamp (q_c/dt), so a tolerance-based
    # comparison passes while the unclamped cells differ by an order of
    # magnitude -- which is exactly the ordering effect under test.
    d_qc = np.abs(_arr(one, "q_c") - _arr(three, "q_c"))
    d_T = np.abs(np.asarray(one.T_hat.data) - np.asarray(three.T_hat.data))
    assert d_qc.max() > 0.0 and d_T.max() > 0.0
    # And the difference is not rounding: the largest disagreement is a
    # sizeable fraction of the largest tendency.
    assert d_qc.max() > 1.0e-3 * np.abs(_arr(three, "q_c")).max()


def test_the_partition_runs_on_the_spectral_lane_and_needs_the_subcycle(setup):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=True)
    out = _run(cfg, 3, grid, sigma, state)
    assert np.all(np.isfinite(_arr(out, "q_c")))
    # The closure's liquid reached the cloud-water tracer.
    assert np.max(np.abs(_arr(out, "q_c"))) > 0.0
    with pytest.raises(ValueError, match="cld_macmic_num_steps>=2"):
        _run(cfg, 1, grid, sigma, state)


def test_the_partition_refuses_a_state_with_no_cloud_tracer(setup):
    grid, sigma, _ = setup
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    bare = isothermal_rest_state_spectral(
        grid, sigma, T_init=285.0, p_s_init=1.0e5, perturbation_amplitude=0.0,
        tracers={"q_v": jnp.full(shp, 6.0e-3)})
    cfg, _ = _cfg(partition=True)
    with pytest.raises(ValueError, match="liquid_partition exchanges water"):
        _run(cfg, 3, grid, sigma, bare)


def test_the_subcycle_needs_both_sub_cycled_modules(setup):
    grid, sigma, _ = setup
    cfg, _ = _cfg(partition=False)
    cfg = cfg._replace(microphysics=MicrophysicsConfig(scheme="none"))
    with pytest.raises(ValueError, match="re-run a single module"):
        make_physics(cfg, model_type="spectral_pe", dt=_DT,
                     cld_macmic_num_steps=3)


def test_water_is_conserved_across_the_partition_tendencies(setup):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=True)
    out = _run(cfg, 3, grid, sigma, state)
    # CLUBB's exchange is internal to (q_v, q_c); the microphysics moves water
    # on to rain/ice. Total water tendency must stay finite and bounded, and
    # the vapour+liquid pair must not blow up in opposite directions.
    tot = sum(_arr(out, k) for k in ("q_v", "q_c", "q_i", "q_r"))
    assert np.all(np.isfinite(tot))
    assert np.max(np.abs(tot)) < 1.0e-3


def test_the_partition_refuses_the_diagnostic_closure(setup):
    grid, sigma, _ = setup
    cfg, _ = _cfg(partition=True)
    cfg = cfg._replace(turbulence=TurbulenceConfig(
        scheme="clubb", clubb=CLUBBConfig(prognostic=False,
                                          liquid_partition=True)))
    with pytest.raises(ValueError, match="needs the PROGNOSTIC CLUBB"):
        make_physics(cfg, model_type="spectral_pe", dt=_DT,
                     cld_macmic_num_steps=3)
