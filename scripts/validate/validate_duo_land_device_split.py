"""Gate: the FV3-duo land step split over N devices (``solve_land_on_devices``)
equals the one-call step to a per-leaf tolerance, on a real land state.

Setup is production's: ``run_amip.main`` with a run's command line plus
``--restart-from <checkpoint>``; only ``ModelDriver.run`` is replaced.  The
inputs are the loop's land call on that checkpoint: the packed land columns,
the checkpoint's pending forcing window meaned as the loop means it, the land
tile's config/params/lat/carbon.  Each N in ``--devices`` runs on the first N
devices; the reference is the same step compiled on one device.  Exit 1 when
any leaf's max|split - one| / max|one| exceeds ``--tol``.

    XLA_FLAGS=--xla_force_host_platform_device_count=6 JAX_PLATFORMS=cpu \\
    python scripts/validate/validate_duo_land_device_split.py \\
        --checkpoint CKPT --tol T --devices 2 6 -- <run_amip args>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def leaf_rel(ref, got) -> dict:
    """``{leaf path: max|got-ref| / max|ref|}`` over floating leaves, finite
    entries only; a NaN-mask mismatch or a nonzero diff on a zero-peak leaf
    is ``inf`` (never passes a tolerance)."""
    import jax
    out = {}
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ref)[0],
                            jax.tree_util.tree_leaves(got)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"{jax.tree_util.keystr(path)}: {a.shape} "
                             f"{a.dtype} vs {b.shape} {b.dtype}")
        if a.dtype.kind != "f":
            out[jax.tree_util.keystr(path)] = (0.0 if np.array_equal(a, b)
                                               else float("inf"))
            continue
        na, nb = np.isnan(a), np.isnan(b)
        fin = ~(na | nb)
        peak = float(np.max(np.abs(a[fin]))) if fin.any() else 0.0
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        rel = d / peak if peak > 0 else (0.0 if d == 0 else float("inf"))
        out[jax.tree_util.keystr(path)] = (float("inf") if np.any(na != nb)
                                           else rel)
    return out


def _gate_run(opts):
    def run(self, start_step: int = 0, start_day=None, **_kw):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh
        from legoesm.driver.model_driver import (
            land_window_from_checkpoint, mpas_land_step_params,
            shard_land_columns, solve_land_on_devices)
        from legoesm.land.multilayer_land import (
            gather_land_columns, step_multilayer_land_with_diagnostics)

        cfg = self.config
        dt = float(cfg.dycore.dt)
        land_k = (max(1, int(round(float(cfg.land_update_seconds) / dt)))
                  if float(getattr(cfg, "land_update_seconds", 0.0)) > 0.0
                  else 1)
        ph = self.physics
        f_land = np.asarray(self._f_land).reshape(-1)
        ncol = int(f_land.shape[0])
        cols = np.nonzero(f_land > 0.0)[0]
        payload = {k: v for k, v in (self._carry_aux or {}).items()
                   if k.startswith(("land_win_", "land_held_"))}
        a2s_sum, n, cz, w, _ = land_window_from_checkpoint(payload, land_k,
                                                           ncol)
        if n == 0:
            raise SystemExit("gate: checkpoint has no pending land window")
        a2s = jax.tree_util.tree_map(lambda s: s / n, a2s_sum)
        a2s = a2s._replace(cos_zenith=cz / w)
        doy, _ = self._calendar_for_radiation(float(start_day))
        params = ph.land_ml_params
        if bool(getattr(cfg, "land_params_refresh", False)):
            upd = getattr(ph, "land_ml_params_update", None)
            if upd is None:
                upd = ph.land_ml_params_update_factory()
            year0 = float(getattr(cfg, "start_year", 0.0) or 0.0)
            params = mpas_land_step_params(
                upd, params, self._land_ml_state.theta_soil[:, 0],
                jnp.asarray(doy - 1.0), jnp.asarray(year0 + start_day / 365.0))
        umin = float(getattr(ph, "land_ml_u_min", 1.0))
        dt_land = land_k * dt
        doy = jnp.asarray(doy, jnp.float64)

        def step(st, a, c, p, la, cb, d):
            return step_multilayer_land_with_diagnostics(
                st, a, c, umin, dt_land, lat=la, doy=d, land_params=p,
                carbon_state=cb)

        def packed(idx):
            j = jnp.asarray(idx)
            pk = lambda t: gather_land_columns(t, j, ncol)  # noqa: E731
            carb = ph.land_ml_carbon
            return (pk(self._land_ml_state), pk(a2s), pk(ph.land_ml_cfg),
                    pk(params), pk(ph.land_ml_lat),
                    pk(carb) if carb is not None else None, doy)

        ref = jax.device_get(jax.jit(step)(*packed(cols)))   # device 0
        report, worst = {}, 0.0
        for nd in opts.devices:
            if jax.device_count() < nd:
                raise SystemExit(f"gate: need {nd} devices, have "
                                 f"{jax.device_count()}")
            slots, keep = shard_land_columns(cols, nd)
            mesh = Mesh(np.array(jax.devices()[:nd]), ("land",))
            got = jax.device_get(jax.jit(
                lambda *a: solve_land_on_devices(step, a, mesh, keep,
                                                 slots.size))(*packed(slots)))
            # whole-run counters are recounted on the split side: compare
            # the per-column results and those counts, not the step's own
            # scalars (identical by construction on the one-call side)
            rel = leaf_rel(ref[:3], got[:3])
            rel.update({f"surface.{k}": v for k, v in leaf_rel(
                ref[3]._replace(n_held=None, n_fallback=None,
                                n_fallback_rejected=None),
                got[3]._replace(n_held=None, n_fallback=None,
                                n_fallback_rejected=None)).items()})
            for k in ("n_held", "n_fallback", "n_fallback_rejected"):
                a, b = getattr(ref[3], k), getattr(got[3], k)
                if a is not None:
                    rel[f"surface.{k}"] = (0.0 if int(a) == int(b)
                                           else float("inf"))
            report[nd] = rel
            worst = max(worst, max(rel.values()))
            bad = {k: v for k, v in rel.items() if v > opts.tol}
            print(f"GATE devices={nd}: {len(rel)} leaves, worst "
                  f"{max(rel.values()):.3e} (tol {opts.tol:.1e}) "
                  f"{'PASS' if not bad else 'FAIL ' + json.dumps(bad)}",
                  flush=True)
        if opts.out:
            Path(opts.out).write_text(json.dumps(
                {"tol": opts.tol, "checkpoint": opts.checkpoint,
                 "devices": {str(k): v for k, v in report.items()}},
                indent=1))
        raise SystemExit(0 if worst <= opts.tol else 1)
    return run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: ... --checkpoint C --tol T -- <run_amip args>")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--devices", type=int, nargs="+", default=[2, 6])
    p.add_argument("--out", default=None)
    opts = p.parse_args(argv[:cut])
    sys.path.insert(0, str(Path.cwd()))
    from legoesm.driver.model_driver import ModelDriver
    ModelDriver.run = _gate_run(opts)
    from scripts.run import run_amip
    run_amip.main(argv[cut + 1:] + ["--restart-from", opts.checkpoint])


if __name__ == "__main__":
    main()
