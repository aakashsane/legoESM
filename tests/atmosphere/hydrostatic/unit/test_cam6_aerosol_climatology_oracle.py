"""CAM6 prescribed-aerosol climatology: horizontal mapping vs the gfortran oracle
(verbatim tracer_data / interpolate_data), lininterp_init checks, and the
month-by-month read == all-at-once read identity (user decision 10a).

Oracle cases: ``scripts/validate/cam6_aerosol_tracer_data_oracle_cases.txt``
(8 x 5 grid, 4 levels, 12 records, 9 columns incl. lon 0, last file lon, between
last lon and 2pi, lat +-pi/2).  Tolerance fp64 rtol 1e-12 / atol 1e-30.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import cam6_aerosol_climatology as clim  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cam_lininterp  # noqa: E402

_DIR = Path(__file__).resolve().parents[4] / "scripts" / "validate"
_REAL = os.environ.get("LEGOESM_CAM6_AEROSOL_DIR")


def _cases():
    rows = [np.array([float(x) for x in l.split()])
            for l in (_DIR / "cam6_aerosol_tracer_data_oracle_cases.txt").read_text().splitlines()]
    nlon, nlat, nlev, nrec, ncol, _ = (int(x) for x in rows[0])
    lons, lats, _t, _y, _a, _b, _p0, tlat, tlon, fld, ps = rows[1:12]
    fld = fld.reshape((nlon, nlat, nlev, nrec), order="F").transpose(3, 2, 1, 0)
    ps = ps.reshape((nlon, nlat, nrec), order="F").transpose(2, 1, 0)
    exp = [np.array([float(x) for x in l.split()])
           for l in (_DIR / "cam6_aerosol_tracer_data_oracle_expected.txt").read_text().splitlines()]
    return lons, lats, tlat, tlon, fld, ps, np.stack(exp[:nlev]), exp[nlev]


def test_horizontal_mapping_matches_gfortran_oracle():
    lons, lats, tlat, tlon, fld, ps, fcol_exp, ps_exp = _cases()
    lon_w, lat_w = clim.horizontal_weights(lats, lons, tlat, tlon)
    np.testing.assert_allclose(cam_lininterp.lininterp_2d(fld[0], lon_w, lat_w), fcol_exp, rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(cam_lininterp.lininterp_2d(ps[0], lon_w, lat_w), ps_exp, rtol=1e-12, atol=1e-30)


def test_numpy_bndry_weights_equal_jax():
    x = np.array([-1.5, -0.2, 0.0, 0.7, 1.5])
    y = np.array([-2.0, -1.5, -0.1, 0.0, 0.7, 1.0, 1.5, 3.0])
    for got, want in zip(cam_lininterp.bndry_weights(x, y, True, xp=np), cam_lininterp.bndry_weights(x, y, True)):
        np.testing.assert_array_equal(got, np.asarray(want))


@pytest.mark.parametrize("yin, yout, method, match", [
    ([0.0], [0.0], 1, "at least 2"),
    ([0.0, 2.0, 1.0], [0.5], 1, "Non-monotonic"),
    ([2.0, 1.0, 0.0], [0.5], 1, "decreasing"),
    ([0.0, 1.0, 2.0], [0.5], 0, "not ported"),
    ([0.0, 1.0, 2.0], [0.5], 2, "dyinwrap"),          # wrap gap 2pi-2 vs spacing 1
    ([0.0, 1.0, 2.0], [np.nan], 1, "without interp indices"),
])
def test_lininterp_init_raises_where_cam_aborts(yin, yout, method, match):
    with pytest.raises(ValueError, match=match):
        cam_lininterp.lininterp_init(yin, yout, method, 0.0, 2 * np.pi)


def _write_synthetic(path, var, nlev):
    import netCDF4
    rng = np.random.default_rng(7)
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", 36)
        d.createDimension("lev", nlev)
        d.createDimension("lat", 3)
        d.createDimension("lon", 4)
        for c, v in (("lat", [-60.0, 0.0, 60.0]), ("lon", [0.0, 90.0, 180.0, 270.0]),
                     ("lev", np.linspace(3.0, 990.0, nlev))):
            d.createVariable(c, "f8", (c,))[:] = v
        x = d.createVariable(var, "f4", ("time", "lev", "lat", "lon"))
        x[:] = rng.uniform(1e-12, 1e-8, (36, nlev, 3, 4)).astype(np.float32)


def _all_at_once(path, var):
    import netCDF4
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        a = np.asarray(d[var][:], dtype=np.float64).reshape((3, 12) + d[var].shape[1:])
    return (a[0] + a[1] + a[2]) / 3.0


def test_month_by_month_equals_all_at_once_synthetic(tmp_path, monkeypatch):
    name = clim.PPE_TIMESERIES.format(var="so4_a1")
    _write_synthetic(tmp_path / name, "so4_a1", 5)
    monkeypatch.setitem(clim.PPE_SHA256, name, hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
    got = clim.read_monthly_climatology(tmp_path, "so4_a1")[0]
    np.testing.assert_array_equal(got, _all_at_once(tmp_path / name, "so4_a1"))


def test_sha_mismatch_raises(tmp_path):
    name = clim.PPE_TIMESERIES.format(var="so4_a1")
    _write_synthetic(tmp_path / name, "so4_a1", 2)
    with pytest.raises(ValueError, match="sha256"):
        clim.read_monthly_climatology(tmp_path, "so4_a1")


@pytest.mark.skipif(not _REAL, reason="set LEGOESM_CAM6_AEROSOL_DIR to the PPE data folder")
def test_real_files_month_by_month_identity_and_shared_grid():
    hyam, hybm, p0 = clim.read_hybrid_coefficients(_REAL)
    assert p0 == 1.0e5
    ref = None
    for var in ("PS",) + clim.PPE_SPECIES_FIELDS + clim.PPE_NUMBER_FIELDS:
        got, lat, lon, lev = clim.read_monthly_climatology(_REAL, var)
        np.testing.assert_array_equal(got, _all_at_once(Path(_REAL) / clim.PPE_TIMESERIES.format(var=var), var))
        import netCDF4
        with netCDF4.Dataset(Path(_REAL) / clim.PPE_TIMESERIES.format(var=var)) as d:
            t = np.asarray(d["time"][:])
        if ref is None:
            ref = (lat, lon, t)
        np.testing.assert_array_equal(lat, ref[0])
        np.testing.assert_array_equal(lon, ref[1])
        np.testing.assert_array_equal(t, ref[2])
        if lev is not None:
            np.testing.assert_array_equal(lev, 1000.0 * (hyam + hybm))
            assert np.all(np.diff(lev) > 0)
    assert len(clim.PPE_SPECIES_FIELDS) == 15 and len(clim.PPE_NUMBER_FIELDS) == 4
