"""Energy the dry-mass fixer changes OUTSIDE the dycore energy-fixer window
(review 2026-10-14 F15 follow-up V8), measured on a production checkpoint
with the full production step (dynamics + energy fixer + physics + rails).

Restores a checkpoint through the production driver and runs N steps of the
real loop.  Two operators are instrumented (jax.debug.callback, numbers only):
  efix    : energy the dycore energy fixer returns   = c_pd * dT * M   [J]
  massfix : energy change across the dry-mass fixer  = E(after) - E(before) [J]
with E = sum_area((c_pd T + KE) dp/g + phis p_s/g), the fixer's own energy
(``dycore_column_energy``; TRiSK KE on MPAS, cell-centre KE on lat-lon).
On lat-lon the mass fixer sits inside the safety rails; its share is
E(rails, fix_mass on) - E(rails, fix_mass off) on the same input.
Rates are W/m^2 = dE / (A * dt), + = energy added.

  --lane mpas   : ``--argv-file`` = run_amip CLI (deck + data paths) of the run
  --lane latlon : ``--config-json`` = the run's experiment_config.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("WT", ".")).resolve()
ROWS: list[tuple[str, float, float]] = []


def _rec(tag):
    def f(de, dps):
        ROWS.append((tag, float(de), float(dps)))
    return f


def _driver_mpas(a, days):
    spec = importlib.util.spec_from_file_location(
        "run_amip_probe", ROOT / "scripts" / "run" / "run_amip.py")
    ra = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_probe"] = ra
    spec.loader.exec_module(ra)
    argv = open(a.argv_file).read().split() + ["--output", a.out]
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    deck = argv[argv.index("--config") + 1]
    deck = str(ROOT / deck) if not os.path.isabs(deck) else deck
    argv[argv.index("--config") + 1] = deck
    keys = load_yaml_config(deck, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    # the checkpoint carries the land state (the runs had land_ic_path="")
    args.land_ic = ""
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    cfg = ra.build_config_from_args(args)
    return cfg._replace(days=days, output=cfg.output._replace(output_dir=a.out))


def _driver_json(a, days):
    from legoesm.driver.config import load_experiment_config
    cfg = load_experiment_config(a.config_json)
    return cfg._replace(days=days, output=cfg.output._replace(output_dir=a.out))


def _instrument_mpas():
    import jax
    import jax.numpy as jnp

    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pm
    from legoesm import constants
    from legoesm.core.conservation import dycore_column_energy
    from legoesm.core.operators_voronoi import kinetic_energy_cell_3d

    def energy(s, mesh, sig):
        ph = sig.pressure_at_half(s.p_s.data)
        e = dycore_column_energy(s.T.data, kinetic_energy_cell_3d(s.u.data, mesh),
                                 ph[..., 1:] - ph[..., :-1], s.phis.data, s.p_s.data)
        return jnp.sum(e * mesh.areaCell)

    orig_mf, orig_ef = pm._fix_mass_mpas_hydro, pm.voronoi_energy_fixer_dT

    def mf(state_new, state_old, mesh, total_area=None, target_mass=None,
           sigma_coord=None):
        out = orig_mf(state_new, state_old, mesh, total_area=total_area,
                      target_mass=target_mass, sigma_coord=sigma_coord)
        a = mesh.areaCell
        dps = jnp.sum((out.p_s.data - state_new.p_s.data) * a) / jnp.sum(a)
        jax.debug.callback(_rec("massfix"),
                           energy(out, mesh, sigma_coord)
                           - energy(state_new, mesh, sigma_coord), dps)
        return out

    def ef(pre, post, mesh, sigma_coord, area, sum_fn=None):
        dT = orig_ef(pre, post, mesh, sigma_coord, area, sum_fn)
        ph = sigma_coord.pressure_at_half(post.p_s.data)
        m = jnp.sum(jnp.sum(ph[..., 1:] - ph[..., :-1], axis=-1) * area) / constants.g
        jax.debug.callback(_rec("efix"), constants.c_pd * dT * m, dT)
        return dT

    pm._fix_mass_mpas_hydro, pm.voronoi_energy_fixer_dT = mf, ef


def _instrument_latlon():
    import jax
    import jax.numpy as jnp

    import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pl
    from legoesm import constants
    from legoesm.core.conservation import dycore_column_energy

    def energy(s, grid, sig):
        ph = sig.pressure_at_half(s.p_s)
        uc, vc = pl._face_to_cell_u(s.u), pl._face_to_cell_v(s.v)
        e = dycore_column_energy(s.T, 0.5 * (uc * uc + vc * vc),
                                 ph[..., 1:] - ph[..., :-1], s.phis, s.p_s)
        return jnp.sum(e * grid.area)

    M = pl.CGridLatLonPrimitiveEquationModel
    orig_rails, orig_ef = M._apply_safety_rails, pl.gridded_energy_fixer_dT

    def rails(self, state, target_mass=None, pre_state=None, *, grid=None,
              sigma_coord=None):
        g = self.grid if grid is None else grid
        sc = self.sigma_coord if sigma_coord is None else sigma_coord
        on = orig_rails(self, state, target_mass, pre_state, grid=grid,
                        sigma_coord=sigma_coord)
        cfg = self.config
        self.config = cfg._replace(fix_mass=False)
        try:
            off = orig_rails(self, state, target_mass, pre_state, grid=grid,
                             sigma_coord=sigma_coord)
        finally:
            self.config = cfg
        dps = jnp.sum((on.p_s - off.p_s) * g.area) / jnp.sum(g.area)
        jax.debug.callback(_rec("massfix"), energy(on, g, sc) - energy(off, g, sc),
                           dps)
        return on

    def ef(pre, post, grid, sigma_coord):
        dT = orig_ef(pre, post, grid, sigma_coord)
        ph = sigma_coord.pressure_at_half(post[3])
        m = jnp.sum(jnp.sum(ph[..., 1:] - ph[..., :-1], axis=-1) * grid.area) / constants.g
        jax.debug.callback(_rec("efix"), constants.c_pd * dT * m, dT)
        return dT

    M._apply_safety_rails, pl.gridded_energy_fixer_dT = rails, ef


def _convert_p3_latlon(src, out_dir):
    """Re-key a fu/composable-p3 lat-lon column checkpoint (``llc_u_face`` /
    ``llc_v_face`` + flattened column fields) into main's ``latloncol_ckpt_v1``
    schema.  Same data: C-grid face winds as stored, cell fields reshaped
    row-major (n_lat, n_lon) -- the ColumnMesh order."""
    d = dict(np.load(src, allow_pickle=False))
    nlat, nlon1, nlev = d["llc_u_face"].shape
    nlon = nlon1 - 1
    to2 = lambda x: x.reshape((nlat, nlon) + x.shape[1:])  # noqa: E731
    d["latloncol_schema"] = np.asarray("latloncol_ckpt_v1")
    d["latloncol_u"], d["latloncol_v"] = d.pop("llc_u_face"), d.pop("llc_v_face")
    for k in ("T", "p_s", "phis"):
        d[f"latloncol_{k}"] = to2(d[k])
    names = sorted(str(n) for n in d["tracer_names"])
    d["latloncol_tracer_names"] = np.asarray(names)
    for i, nm in enumerate(names):
        d[f"latloncol_q_{i}"] = to2(d[f"trc_{nm}"])
    out = os.path.join(out_dir, "converted_" + os.path.basename(src))
    np.savez(out, **d)
    print(f"converted {src} -> {out} (nlat={nlat} nlon={nlon} nlev={nlev})",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=("mpas", "latlon"), required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--argv-file", default="")
    ap.add_argument("--config-json", default="")
    ap.add_argument("--convert-p3", action="store_true")
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"WT={ROOT} sha={sha} lane={a.lane} restart={a.restart} "
          f"steps={a.steps}", flush=True)
    {"mpas": _instrument_mpas, "latlon": _instrument_latlon}[a.lane]()
    if a.lane == "mpas":
        cfg = _driver_mpas(a, 1.0)
    else:
        cfg = _driver_json(a, 1.0)
    dt = float(cfg.dycore.dt)
    cfg = cfg._replace(days=(a.steps + 0.5) * dt / 86400.0)
    print(f"dt={dt} fix_mass={cfg.dycore.fix_mass} "
          f"energy_fixer={cfg.dycore.energy_fixer} grid={cfg.grid.grid_type} "
          f"res={cfg.grid.resolution} column_lane={cfg.dycore.column_lane}",
          flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.convert_p3:
        a.restart = _convert_p3_latlon(a.restart, a.out)
    from legoesm.driver.model_driver import ModelDriver
    drv = ModelDriver(cfg)
    drv.setup()
    if a.convert_p3:
        # The P3 run's terrain product predates main's; the restore refuses a
        # different terrain.  Adopt the CHECKPOINT's phis (the dynamics state
        # is balanced with it) in the fresh state the restore compares to.
        import jax.numpy as jnp
        ck = np.load(a.restart)["latloncol_phis"]
        nat = drv.state.native
        print(f"adopting checkpoint terrain: max|dphis|="
              f"{float(np.max(np.abs(np.asarray(nat.phis) - ck))):.3e} m2/s2",
              flush=True)
        drv.state = drv.model.from_bundle(
            nat._replace(phis=jnp.asarray(ck, dtype=nat.phis.dtype)))
        if getattr(drv, "_phis_data", None) is not None:
            drv._phis_data = jnp.reshape(jnp.asarray(ck), drv._phis_data.shape)
    step0, day0 = drv.load_checkpoint(a.restart)
    print(f"restored step={step0} day={day0} model={type(drv.model).__name__}",
          flush=True)
    area = np.asarray(drv.grid.areaCell if hasattr(drv.grid, "areaCell")
                      else drv.grid.area)
    A = float(area.sum())
    drv.run(start_step=step0, start_day=day0)
    ef = [r for r in ROWS if r[0] == "efix"]
    mf = [r for r in ROWS if r[0] == "massfix"]
    print(f"A={A:.6e} n_efix={len(ef)} n_massfix={len(mf)}", flush=True)
    for i, (e, m) in enumerate(zip(ef, mf)):
        print(f"step {i + 1:3d} efix {e[1] / (A * dt):+.6e} W/m2 (dT={e[2]:+.3e} K)"
              f"  massfix {m[1] / (A * dt):+.6e} W/m2 (dps={m[2]:+.3e} Pa)",
              flush=True)
    efr = np.array([e[1] for e in ef]) / (A * dt)
    mfr = np.array([m[1] for m in mf]) / (A * dt)
    dps = np.array([m[2] for m in mf])
    print(json.dumps({
        "lane": a.lane, "restart": a.restart, "sha": sha, "dt": dt,
        "steps": len(mf), "efix_mean_Wm2": float(efr.mean()),
        "massfix_mean_Wm2": float(mfr.mean()),
        "massfix_absmean_Wm2": float(np.abs(mfr).mean()),
        "massfix_min_Wm2": float(mfr.min()), "massfix_max_Wm2": float(mfr.max()),
        "dps_absmean_Pa": float(np.abs(dps).mean()),
        "ratio_massfix_over_efix": float(mfr.mean() / efr.mean())}), flush=True)


if __name__ == "__main__":
    main()
