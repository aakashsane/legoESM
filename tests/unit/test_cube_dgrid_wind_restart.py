"""Cube D-grid winds: checkpoint round trip.

With persistent D-grid winds the checkpoint saves the corner winds next to
the cell-centre state, so a restarted run continues from exactly the winds
the straight run carried; a checkpoint written before they were saved is
lifted once, with a warning.
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    FV3HydrostaticState)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver


def _cfg(out, days):
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid",
                            dt=1200.0, fix_mass=True, persistent_dgrid=True),
        output=OutputConfig(output_dir=str(out), diag_days=0,
                            checkpoint_days=1),
        days=days, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, distributed=False, n_devices=1,
        surface_bulk_scheme="none")


def _winds(d):
    return tuple(np.asarray(getattr(d.state, f).data, np.float64)
                 for f in ("u_d", "v_d"))


def _resume(tmp, ckpt, tag):
    d = ModelDriver(_cfg(tmp / tag, 2))
    d.setup()
    d.load_checkpoint(ckpt)
    d._enter_persistent_dgrid()
    return d


@pytest.fixture(scope="module")
def leg1(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cubewind")
    b1 = ModelDriver(_cfg(tmp / "leg1", 1))
    b1.setup()
    assert b1.run() == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp / "leg1" / "checkpoint_day_*.npz")))[-1]
    return tmp, b1, ckpt


def test_restart_resumes_the_saved_corner_winds(leg1):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        assert any("cube_u_d" in k for k in z.keys())
    d = _resume(tmp, ckpt, "leg2")
    assert isinstance(d.state, FV3HydrostaticState)
    for x, y in zip(_winds(b1), _winds(d)):
        np.testing.assert_array_equal(x, y)
    assert not any(k in d._carry_aux for k in d._CORNER_WIND_KEYS)


def test_two_legs_equal_the_straight_run(leg1):
    tmp, _b1, ckpt = leg1
    straight = ModelDriver(_cfg(tmp / "straight", 2))
    straight.setup()
    assert straight.run() == "COMPLETED"
    leg2 = ModelDriver(_cfg(tmp / "leg2_run", 2))
    leg2.setup()
    step, day = leg2.load_checkpoint(ckpt)
    assert leg2.run(start_step=step, start_day=day) == "COMPLETED"
    for x, y in zip(_winds(straight), _winds(leg2)):
        np.testing.assert_array_equal(x, y)


def test_old_checkpoint_is_lifted_once_with_a_warning(leg1, caplog):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys() if "cube_" not in k}
    old = tmp / "old_checkpoint_day_0001.npz"
    np.savez(old, **data)
    with caplog.at_level(logging.WARNING):
        d = _resume(tmp, str(old), "leg2_old")
    assert "no saved D-grid winds" in caplog.text
    assert np.abs(_winds(b1)[0] - _winds(d)[0]).max() > 0.0


@pytest.mark.parametrize("corrupt", [
    lambda a: a * 1.1,
    lambda a: np.roll(a, 1, axis=-2),
])
def test_corner_winds_from_another_state_are_refused(leg1, corrupt):
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()}
    key = next(k for k in data if "cube_u_d" in k)
    data[key] = corrupt(data[key])
    bad = tmp / "bad_checkpoint_day_0001.npz"
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_bad", 2))
    d.setup()
    d.load_checkpoint(str(bad))
    for _ in range(2):          # a retried entry must refuse again, not lift
        with pytest.raises(ValueError, match="do not average"):
            d._enter_persistent_dgrid()
        assert all(k in d._carry_aux for k in d._CORNER_WIND_KEYS)
