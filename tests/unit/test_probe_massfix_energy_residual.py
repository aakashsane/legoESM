"""Direct test for ``scripts/validate/probe_massfix_energy_residual.py``
(review 2026-10-14 F15 V8).

The probe's numbers went into the vote, so its one piece of data handling is
pinned: re-keying a fu/composable-p3 lat-lon column checkpoint into main's
``latloncol_ckpt_v1`` schema keeps every value and reshapes the flattened
column fields row-major onto ``(n_lat, n_lon)`` (the ColumnMesh order).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "probe_massfix_energy_residual.py"


def _load():
    spec = importlib.util.spec_from_file_location("_massfix_probe", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_massfix_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_p3_checkpoint_is_rekeyed_without_changing_values(tmp_path):
    nlat, nlon, nlev = 3, 4, 2
    rng = np.random.default_rng(0)
    T = rng.random((nlat * nlon, nlev))
    p_s, phis = rng.random(nlat * nlon), rng.random(nlat * nlon)
    q_v, q_c = rng.random((nlat * nlon, nlev)), rng.random((nlat * nlon, nlev))
    u_f, v_f = rng.random((nlat, nlon + 1, nlev)), rng.random((nlat + 1, nlon, nlev))
    src = tmp_path / "ck.npz"
    np.savez(src, T=T, p_s=p_s, phis=phis, trc_q_v=q_v, trc_q_c=q_c,
             tracer_names=np.asarray(["q_v", "q_c"]), llc_u_face=u_f,
             llc_v_face=v_f)
    out = np.load(_load()._convert_p3_latlon(str(src), str(tmp_path)))
    assert str(out["latloncol_schema"]) == "latloncol_ckpt_v1"
    np.testing.assert_array_equal(out["latloncol_u"], u_f)
    np.testing.assert_array_equal(out["latloncol_v"], v_f)
    np.testing.assert_array_equal(out["latloncol_T"], T.reshape(nlat, nlon, nlev))
    np.testing.assert_array_equal(out["latloncol_p_s"], p_s.reshape(nlat, nlon))
    np.testing.assert_array_equal(out["latloncol_phis"], phis.reshape(nlat, nlon))
    # tracers sorted by name, slot i = i-th name
    assert list(out["latloncol_tracer_names"]) == ["q_c", "q_v"]
    np.testing.assert_array_equal(out["latloncol_q_0"], q_c.reshape(nlat, nlon, nlev))
    np.testing.assert_array_equal(out["latloncol_q_1"], q_v.reshape(nlat, nlon, nlev))
    # row-major: column k = (k // nlon, k % nlon)
    assert out["latloncol_p_s"][1, 2] == p_s[1 * nlon + 2]
