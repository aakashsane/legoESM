"""The lat-lon C-grid dycore as a column model (``dycore.column_lane`` on
``latlon_cgrid``): the shared column loop drives the native C-grid state
through :class:`LatLonCGridColumnModel`."""
import numpy as np
import pytest

import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.latlon_cgrid_column import (
    LatLonCGridColumnModel, LatLonColumnState, build_latlon_column_mesh)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
    hydrostatic_to_cgrid)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticTendencies

RES, NLEV, DT = 12, 8, 600.0


@pytest.fixture(scope="module")
def setup():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    g = create_grid("latlon", RES)
    s = create_sigma_coordinate(NLEV)
    hs = held_suarez_init_latlon(g, s, T_init=280.0)
    nat = hydrostatic_to_cgrid(hs, g)
    nat = nat._replace(tracers={"q_v": 5e-3 * jnp.ones_like(nat.T)})
    return g, s, nat


def _dyn(g, s):
    cfg = CGridLatLonPrimitiveEquationConfig(use_polar_filter=True,
                                             energy_fixer=True)
    return CGridLatLonPrimitiveEquationModel(g, s, cfg, dt=DT)


def _tend(T, p_s, *, du=1e-4, dT=1e-4, precip=None, lat=None, lon=None):
    z, z2 = jnp.zeros_like(T), jnp.zeros_like(p_s)
    if lat is not None:
        # cell-dependent, so a mis-ordered column<->grid map is visible
        z = z + 1e-4 * (jnp.sin(lat) * jnp.cos(lon))[..., None]
    f = lambda a, n: Field(data=a, name=n)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=f(z + du, "du"), dT_dt=f(z + dT, "dT"), dp_s_dt=f(z2, "dps"),
        dphis_dt=f(z2, "dphis"), dv_dt=f(z, "dv"),
        precip=None if precip is None else f(z2 + precip, "pr"))


def test_mesh_is_the_row_major_cells_of_the_grid(setup):
    g, _, _ = setup
    m = build_latlon_column_mesh(g)
    assert m.nCells == g.n_lat * g.n_lon
    np.testing.assert_array_equal(np.asarray(m.latCell),
                                  np.asarray(g.lat2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.lonCell),
                                  np.asarray(g.lon2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.areaCell),
                                  np.asarray(g.area).reshape(-1))
    lat, lon = np.asarray(m.latCell), np.asarray(m.lonCell)
    clat, clon = np.asarray(m.cornerLat), np.asarray(m.cornerLon)
    assert clat.shape == (m.nCells, 4)
    assert (clat.min(axis=1) < lat).all() and (clat.max(axis=1) > lat).all()
    assert (clon.min(axis=1) < lon).all() and (clon.max(axis=1) > lon).all()


def test_dynamics_only_step_is_the_native_step_bitwise(setup):
    g, s, nat = setup
    ref = _dyn(g, s).step(nat, DT)
    col = LatLonCGridColumnModel(_dyn(g, s))
    out = col.step(col.from_bundle(nat), DT)
    assert isinstance(out, LatLonColumnState)
    for k in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(out.native, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    np.testing.assert_array_equal(np.asarray(out.native.tracers["q_v"]),
                                  np.asarray(ref.tracers["q_v"]))


def test_column_physics_is_the_native_operator_split_bitwise(setup):
    """Column physics through the bridge == the same tendencies given to the
    native step on the grid (same increment applier, polar filter, rails),
    and the column-shaped surface diagnostics are stashed."""
    g, s, nat = setup

    def col_phys(cols, mesh, sigma, phys_state=None, forcing=None):
        assert cols.T.data.shape == (mesh.nCells, NLEV)
        return _tend(cols.T.data, cols.p_s.data, precip=1.0,
                     lat=mesh.latCell, lon=mesh.lonCell)

    def grid_phys(hs, grid, sigma):
        # the mesh holds the grid's coordinates in fp64
        return _tend(hs.T.data, hs.p_s.data,
                     lat=jnp.asarray(grid.lat2d, jnp.float64),
                     lon=jnp.asarray(grid.lon2d, jnp.float64))

    ref = _dyn(g, s).step(nat, DT, physics_fn=grid_phys)
    col = LatLonCGridColumnModel(_dyn(g, s))
    out = col.step(col.from_bundle(nat), DT, physics_fn=col_phys)
    for k in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(out.native, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    # the physics moved the state (the comparison has power)
    noph = _dyn(g, s).step(nat, DT)
    assert np.abs(np.asarray(ref.T) - np.asarray(noph.T)).max() > 0.05
    assert np.abs(np.asarray(ref.u) - np.asarray(noph.u)).max() > 0.05
    assert col._sfc_diag is not None
    assert col._sfc_diag[2].data.shape == (g.n_lat * g.n_lon,)
    assert float(jnp.max(jnp.abs(col._sfc_diag[2].data - 1.0))) == 0.0


def test_refuses_physics_without_the_meridional_tendency(setup):
    g, s, nat = setup

    def no_v(cols, mesh, sigma, phys_state=None, forcing=None):
        return _tend(cols.T.data, cols.p_s.data)._replace(dv_dt=None)

    col = LatLonCGridColumnModel(_dyn(g, s))
    with pytest.raises(ValueError, match="dv_dt=None"):
        col.step(col.from_bundle(nat), DT, physics_fn=no_v)


def test_writes_back_T_and_tracers_and_refuses_wind_edits(setup):
    g, s, nat = setup
    col = LatLonCGridColumnModel(_dyn(g, s))
    st = col.from_bundle(nat)
    edited = st._replace(T=st.T.replace(data=st.T.data + 1.0), tracers={
        "q_v": st.tracers["q_v"].replace(data=st.tracers["q_v"].data * 2.0)})
    back = col.to_bundle(edited)
    np.testing.assert_array_equal(np.asarray(back.T), np.asarray(nat.T) + 1.0)
    np.testing.assert_array_equal(np.asarray(back.tracers["q_v"]),
                                  2.0 * np.asarray(nat.tracers["q_v"]))
    np.testing.assert_array_equal(np.asarray(back.u), np.asarray(nat.u))
    for nm in ("u", "v", "p_s"):
        bad = st._replace(**{nm: getattr(st, nm).replace(
            data=getattr(st, nm).data + 1.0)})
        with pytest.raises(ValueError, match=f"state.{nm} was edited"):
            col.to_bundle(bad)
    with pytest.raises(ValueError, match="not the one this model returned"):
        col.to_bundle(st._replace(native=nat._replace(T=nat.T + 0.0)))


# ---------------------------------------------------------------------
# the driver: _run_column with the lat-lon column model
# ---------------------------------------------------------------------

def _driver_cfg(tmp_path, **over):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    dycore = DycoreConfig(
        model_type="hydrostatic", discretization="latlon_cgrid", dt=DT,
        column_lane=over.pop("column", True), use_polar_filter=True)
    base = dict(
        grid=GridConfig(grid_type="latlon", resolution=RES, nlev=NLEV),
        dycore=dycore, days=over.pop("days", 0.25),
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none", precision="fp64",
        output=OutputConfig(diag_days=0, checkpoint_days=0,
                            output_dir=str(tmp_path)))
    base.update(over)
    return ExperimentConfig(**base)


def _driver(tmp_path, **over):
    from legoesm.driver.model_driver import ModelDriver
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _driver_cfg(tmp_path, **over)
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    return drv


def test_driver_column_lane_steps_the_native_dycore(tmp_path):
    """``run()`` goes through the column loop with the lat-lon column model;
    its native state equals the bare dycore stepped from the same IC.  The
    loop calls its (all-"none") physics package every step, so the two
    programs differ by zero increments through the polar filter and rails:
    equal to 1e-10 absolute (measured 1.2e-12 m/s, 7e-13 K after 36 steps)."""
    drv = _driver(tmp_path)
    assert isinstance(drv.model, LatLonCGridColumnModel)
    assert isinstance(drv.state, LatLonColumnState)
    assert drv.grid is drv.model.mesh and drv.sigma is drv.model.sigma_coord
    x = drv.state.native
    assert drv.run() == "COMPLETED"
    dyn = CGridLatLonPrimitiveEquationModel(
        drv.model.grid, drv.model.sigma_coord, drv.model.config, dt=DT)
    for _ in range(int(round(0.25 * 86400.0 / DT))):
        x = dyn.step(x, DT)
    for k in ("u", "v", "T", "p_s"):
        a, b = np.asarray(getattr(drv.state.native, k)), np.asarray(getattr(x, k))
        assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max()), k
    assert np.abs(np.asarray(x.T) - np.asarray(drv._state_template.native.T)).max() > 0


def test_driver_refusals(tmp_path):
    from legoesm.driver.config import OutputConfig
    from legoesm.driver.model_driver import ModelDriver
    ledger = OutputConfig(diag_days=0, checkpoint_days=0,
                          output_dir=str(tmp_path), budget_ledger=True)
    for over, frag in ((dict(sponge_enabled=True), "sponge_enabled"),
                       (dict(mpas_qv_smooth_del4_m4s=1e14), "Voronoi mesh"),
                       (dict(ensemble_size=2), "single device"),
                       (dict(output=ledger), "budget_ledger"),
                       (dict(convection="kuo"), "grid operator")):
        drv = ModelDriver(_driver_cfg(tmp_path, **over), output_dir=tmp_path)
        with pytest.raises(ValueError, match=frag):
            drv.setup()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("phys", [dict(), dict(microphysics="kessler")])
def test_driver_restart_is_bitwise(tmp_path, phys):
    """Run A: 2 days with a checkpoint each day; run B: a fresh driver from
    A's day-1 checkpoint for the remaining day.  The native C-grid states
    (face winds included) must be BITWISE equal."""
    from legoesm.driver.model_driver import ModelDriver
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(), dir_b.mkdir()
    cfg_a = _driver_cfg(dir_a, days=2, **phys)
    cfg_a = cfg_a._replace(output=cfg_a.output._replace(checkpoint_days=1))
    drv_a = ModelDriver(cfg_a, output_dir=dir_a)
    drv_a.setup()
    assert drv_a.run() == "COMPLETED"
    mid = dir_a / "checkpoint_day_0001.npz"
    with np.load(mid) as d:
        assert str(d["latloncol_schema"]) == "latloncol_ckpt_v1"
        u_mid = np.asarray(d["latloncol_u"])
    drv_b = ModelDriver(_driver_cfg(dir_b, days=1, **phys), output_dir=dir_b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(mid)
    assert day == pytest.approx(1.0)
    assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"
    a, b = drv_a.state.native, drv_b.state.native
    for k in ("u", "v", "T", "p_s", "phis"):
        assert _same(getattr(a, k), getattr(b, k)), k
    assert sorted(a.tracers) == sorted(b.tracers)
    if phys:
        # the moist run carries its water on the dycore state
        assert "q_v" in a.tracers and float(np.max(a.tracers["q_v"])) > 0.0
    for nm in a.tracers:
        assert _same(a.tracers[nm], b.tracers[nm]), nm
    assert not _same(u_mid, a.u)        # the second day moved the state
    # the driver's lat-lon dycore fixes dry mass step to step (no anchor:
    # the factory never sets anchor_mass_to_initial), so nothing to restore
    assert drv_a.model._target_mass is None and drv_b.model._target_mass is None


def test_driver_refuses_a_checkpoint_without_the_cgrid_state(tmp_path):
    drv = _driver(tmp_path)
    bad = tmp_path / "checkpoint_day_0000.npz"
    np.savez(bad, u=np.zeros(1), T=np.zeros(1), p_s=np.zeros(1),
             phis=np.zeros(1), step=np.asarray(3), day=np.asarray(0.1))
    with pytest.raises(ValueError, match="carries no lat-lon C-grid state"):
        drv.load_checkpoint(bad)
    np.savez(bad, u=np.zeros(1), T=np.zeros(1), p_s=np.zeros(1),
             phis=np.zeros(1), step=np.asarray(3), day=np.asarray(0.1),
             latloncol_schema=np.asarray("latloncol_ckpt_v0"))
    with pytest.raises(ValueError, match="unknown lat-lon column schema"):
        drv.load_checkpoint(bad)


def test_driver_calibrated_land_and_prognostic_clubb_run(tmp_path, monkeypatch):
    """The two configurations the PhysicsPipeline lat-lon lane refuses
    validate and run on the lat-lon column lane: the multilayer land's
    calibrated physics with its solved-flux handoff, and CLUBB carrying its
    moments in the physics state."""
    from legoesm.land.config import biophysics_lmip_two_leaf_setup
    from tests._land_surfdata import require_surfdata
    from tests.unit.test_mpas_multilayer_land_port import _patch_land_loaders
    _patch_land_loaders(monkeypatch)
    cal = biophysics_lmip_two_leaf_setup()["soil_grid"]
    over = dict(
        days=1200.0 / 86400.0, dataset="analytical", radiation="gray",
        turbulence="clubb", clubb_prognostic=True,
        land_mask_path="synthetic.nc", use_multilayer_land=True,
        multilayer_n_layers=cal.n_layers,
        multilayer_soil_depth=cal.total_depth, land_beta_soil=True,
        land_surface_scheme="two_leaf", land_stomatal_beta=False,
        land_calibrated_physics=True, snow_albedo_feedback=True,
        surfdata_path=require_surfdata())
    _driver_cfg(tmp_path, **dict(over)).validate_strict()
    with pytest.raises(ValueError, match="land_calibrated_physics=True is "
                                         "supported only on the MPAS"):
        _driver_cfg(tmp_path, column=False, **dict(over)).validate_strict()
    with pytest.raises(ValueError, match="clubb_prognostic=True is not wired"):
        _driver_cfg(tmp_path, column=False,
                    **{**over, "land_calibrated_physics": False}
                    ).validate_strict()
    drv = _driver(tmp_path, **over)
    assert drv.run() == "COMPLETED"
    ps = drv._mpas_phys_state
    assert ps is not None and ps.clubb_moments is not None
    mom = np.asarray(ps.clubb_moments)
    assert mom.shape[0] == RES * 2 * RES and np.isfinite(mom).all()
    assert np.abs(mom).max() > 0.0


def test_driver_terrain_is_the_lat_lon_lanes_product(tmp_path):
    """The column lane builds the terrain product (binning + the lat-lon
    neighbour filter) on the NATIVE grid: phis and the land fraction equal
    the PhysicsPipeline lat-lon lane's, flattened row-major, and the IC's
    surface geopotential is that terrain."""
    from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (
        _write_elevation_file)
    elev = tmp_path / "elev.nc"
    _write_elevation_file(elev)
    over = dict(topography=str(elev), topo_smoothing=2)
    col = _driver(tmp_path / "col", **over)
    pipe = _driver(tmp_path / "pipe", column=False, **over)
    p_pipe = np.asarray(pipe._phis_data).reshape(-1)
    assert np.abs(p_pipe).max() > 1000.0
    np.testing.assert_array_equal(np.asarray(col._phis_data), p_pipe)
    np.testing.assert_array_equal(np.asarray(col._f_land),
                                  np.asarray(pipe._f_land).reshape(-1))
    np.testing.assert_array_equal(np.asarray(col.state.native.phis),
                                  np.asarray(pipe._phis_data))


def test_ice_curve_drain_is_admitted_on_the_column_lane(tmp_path):
    """The ice-curve drain lives in the column loop's post-step hook, so the
    lat-lon column lane accepts it; the PhysicsPipeline lat-lon lane, where
    it would be inert, still refuses it."""
    over = dict(microphysics="morrison", hard_saturation_adjustment=True,
                hard_sat_ice_curve=True)
    _driver_cfg(tmp_path, **over).validate_strict()
    with pytest.raises(ValueError, match="requires the column loop"):
        _driver_cfg(tmp_path, column=False, **over).validate_strict()


def test_driver_restart_carries_clubb_moments_and_land(tmp_path, monkeypatch):
    """Restart with the two carries this lane exists for: CLUBB's prognostic
    moments (PhysicsState) and the multilayer land state.  Run A: 2 days,
    checkpoint each day; run B from A's day-1 checkpoint.  Native state and
    CLUBB moments must be bitwise equal, and the moments must have evolved
    after the checkpoint (else the comparison has no power)."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.land.config import biophysics_lmip_two_leaf_setup
    from tests._land_surfdata import require_surfdata
    from tests.unit.test_mpas_multilayer_land_port import _patch_land_loaders
    _patch_land_loaders(monkeypatch)
    cal = biophysics_lmip_two_leaf_setup()["soil_grid"]
    over = dict(
        dataset="analytical", radiation="gray", turbulence="clubb",
        clubb_prognostic=True, land_mask_path="synthetic.nc",
        use_multilayer_land=True, multilayer_n_layers=cal.n_layers,
        multilayer_soil_depth=cal.total_depth, land_beta_soil=True,
        land_surface_scheme="two_leaf", land_stomatal_beta=False,
        land_calibrated_physics=True, snow_albedo_feedback=True,
        surfdata_path=require_surfdata())
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(), dir_b.mkdir()
    cfg_a = _driver_cfg(dir_a, days=2, **over)
    cfg_a = cfg_a._replace(output=cfg_a.output._replace(checkpoint_days=1))
    drv_a = ModelDriver(cfg_a, output_dir=dir_a)
    drv_a.setup()
    assert drv_a.run() == "COMPLETED"
    mid = dir_a / "checkpoint_day_0001.npz"
    with np.load(mid) as d:
        mom_mid = np.asarray(d["physstate_clubb_moments"])
    drv_b = ModelDriver(_driver_cfg(dir_b, days=1, **over), output_dir=dir_b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(mid)
    assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"
    a, b = drv_a.state.native, drv_b.state.native
    for k in ("u", "v", "T", "p_s"):
        assert _same(getattr(a, k), getattr(b, k)), k
    for nm in a.tracers:
        assert _same(a.tracers[nm], b.tracers[nm]), nm
    mom_a = np.asarray(drv_a._mpas_phys_state.clubb_moments)
    assert _same(mom_a, drv_b._mpas_phys_state.clubb_moments)
    assert mom_mid.tobytes() != mom_a.tobytes()
