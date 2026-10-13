"""Lat-lon C-grid face winds: checkpoint round trip and the lanes that refuse.

Checkpoints save the native face winds next to the cell-centre state, so a
restarted run continues exactly; a checkpoint written before they were saved
is lifted once (with a warning) and is close to, not equal to, the straight
run.  Lanes that cannot carry the face winds refuse to run.
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    LatLonFaceWindState)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

_DT = 600.0


def _cfg(out, days, **kw):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT,
                            fix_mass=True),
        output=OutputConfig(output_dir=str(out), diag_days=0,
                            checkpoint_days=1),
        days=days, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, distributed=False, n_devices=1,
        surface_bulk_scheme="none", **kw)


def _winds(d):
    return (np.asarray(d.state.u_f.data, np.float64),
            np.asarray(d.state.v_f.data, np.float64))


def _resume(tmp_path, ckpt, tag):
    """Load ``ckpt`` and enter the face-wind lane exactly as ``run`` does."""
    d = ModelDriver(_cfg(tmp_path / tag, 2))
    d.setup()
    d.load_checkpoint(ckpt)
    d._enter_persistent_cgrid()
    return d


@pytest.fixture(scope="module")
def leg1(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("facewind")
    b1 = ModelDriver(_cfg(tmp / "leg1", 1))
    b1.setup()
    assert b1.run(compiled=True) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp / "leg1" / "checkpoint_day_*.npz")))[-1]
    return tmp, b1, ckpt


def test_checkpoint_saves_face_winds_and_restart_resumes_them(leg1):
    """The resumed run starts from the SAVED face winds, not from a lift of
    the cell-centre winds.  (Whole-run restart equality is not asserted: the
    lat-lon compiled lane is not restart-exact even without face winds.)"""
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        assert any("latlon_u_face" in k for k in z.keys())
    d = _resume(tmp, ckpt, "leg2")
    assert isinstance(d.state, LatLonFaceWindState)
    for x, y in zip(_winds(b1), _winds(d)):
        np.testing.assert_array_equal(x, y)


def test_old_checkpoint_is_lifted_once_with_a_warning(leg1, caplog):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys() if "latlon_" not in k}
    old = tmp / "old_checkpoint_day_0001.npz"
    np.savez(old, **data)
    with caplog.at_level(logging.WARNING):
        d = _resume(tmp, str(old), "leg2_old")
    assert "no saved face winds" in caplog.text
    # A lift of the averaged winds cannot give back the carried faces.
    du = np.abs(_winds(b1)[0] - _winds(d)[0]).max()
    assert du > 0.0, du


def test_unsupported_lanes_refuse(tmp_path):
    ens = ModelDriver(_cfg(tmp_path / "ens", 1, ensemble_size=2))
    ens.setup()
    with pytest.raises(SystemExit, match="ensemble"):
        ens.run(compiled=True)

    cpl = ModelDriver(_cfg(tmp_path / "cpl", 1))
    cpl.setup()
    cpl._requires_surface_flux_export = True
    with pytest.raises(SystemExit, match="COUPLED"):
        cpl.run(compiled=True)

    trn = ModelDriver(_cfg(tmp_path / "trn", 1))
    trn.setup()
    with pytest.raises(NotImplementedError, match="face"):
        trn.build_training_segment(4)


def test_checkpoint_with_one_face_wind_is_refused(leg1):
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys() if "latlon_v_face" not in k}
    half = tmp / "half_checkpoint_day_0001.npz"
    np.savez(half, **data)
    with pytest.raises(ValueError, match="only one of the face winds"):
        _resume(tmp, str(half), "leg2_half")
