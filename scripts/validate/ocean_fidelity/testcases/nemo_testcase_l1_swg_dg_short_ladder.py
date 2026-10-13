#!/usr/bin/env python3
"""Score 1--10-step legoESM states against single-rank NEMO restarts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402

CHECKOUT_IMPORT = bootstrap_checkout(__file__)
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_trajectory(restarts, trajectory, masks, *, exact_row) -> list[dict]:
    """Return one exact-comparison row per step and prognostic field."""
    rows = []
    mapping = (("sshn", "ssh_m", "T"), ("un", "u_m_s", "U"),
               ("vn", "v_m_s", "V"))
    for step, candidate in enumerate(trajectory, start=1):
        for nemo_name, lego_name, grid in mapping:
            row = exact_row(
                restarts[step][nemo_name], candidate[lego_name], masks[grid],
                name=f"step{step}.{lego_name}:{nemo_name}")
            row.update({"step": step, "field": lego_name,
                        "nemo_field": nemo_name, "grid": grid})
            rows.append(row)
    return rows


def _configure_backend(backend: str, gpu_device: str | None) -> None:
    if backend == "gpu":
        require(gpu_device is not None and gpu_device.isdigit(),
                "GPU ladder requires --gpu-device with a physical index")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
        os.environ["JAX_PLATFORMS"] = "gpu"
    else:
        require(gpu_device is None, "--gpu-device is valid only with --backend gpu")
        os.environ["JAX_PLATFORMS"] = "cpu"


def run(root: Path, *, composition: str, backend: str,
        gpu_device: str | None, plant: bool = True) -> dict:
    _configure_backend(backend, gpu_device)
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.provenance import git_sha

    legoesm_sha = git_sha()
    require(jax.default_backend() == backend,
            f"requested {backend}, JAX selected {jax.default_backend()}")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(
        transcendentals="native" if backend == "gpu" else "libm"))
    ladder = _load("swg_dg_short_ladder_base",
                   HERE / "nemo_testcase_l1_swg_ladder.py")
    round7 = _load("swg_dg_short_ladder_round7",
                   HERE / "nemo_testcase_l1_swg_dg_round7.py")
    composition_tool = _load(
        "swg_dg_short_ladder_composition",
        HERE / "nemo_testcase_l1_swg_dg_composition.py")
    driver = _load("swg_dg_short_ladder_driver",
                   REPO_ROOT / "scripts/run/run_swg_double_gyre.py")
    namelist = ladder._deck_namelist("maddison")  # noqa: SLF001
    card, masks, model, forcing = ladder._swg_production_case(  # noqa: SLF001
        "short restart ladder", namelist=namelist)
    shape = tuple(card.recipe.land_mask.shape)
    restarts = {
        step: composition_tool._load_single_rank_restart(  # noqa: SLF001
            root, step, shape)[0]
        for step in range(1, 11)
    }
    state = driver.seed_steady_forcing_carry(card.recipe.initial_state, forcing)
    trajectory = []
    if composition == "outer-scan":
        def body(carry, _):
            carry = model.step(carry, card.dt_s, surface_forcing=forcing)
            fields = (carry.eta.data, carry.u.data[:, 1:, 0],
                      carry.v.data[1:, :, 0])
            return carry, fields

        @jax.jit
        def scanned(carry):
            return jax.lax.scan(body, carry, None, length=10)

        _, arrays = scanned(state)
        jax.block_until_ready(arrays[0])
        for index in range(10):
            trajectory.append({
                "ssh_m": np.asarray(arrays[0][index]),
                "u_m_s": np.asarray(arrays[1][index]),
                "v_m_s": np.asarray(arrays[2][index]),
            })
    else:
        for _ in range(10):
            state = model.step(state, card.dt_s, surface_forcing=forcing)
            jax.block_until_ready(state.eta.data)
            trajectory.append({
                "ssh_m": np.asarray(state.eta.data),
                "u_m_s": np.asarray(state.u.data[:, 1:, 0]),
                "v_m_s": np.asarray(state.v.data[1:, :, 0]),
            })

    def exact(reference, candidate, mask, *, name):
        return round7.exact_row(
            reference, candidate, mask, name=name, ladder=ladder)

    rows = score_trajectory(restarts, trajectory, masks, exact_row=exact)
    control = None
    if plant:
        moved = trajectory[0]["u_m_s"].copy()
        wet = np.argwhere(masks["U"] & (moved != 0.0))
        require(wet.size > 0, "one-ULP control has no nonzero wet U")
        at = tuple(wet[0])
        moved[at] = np.nextafter(moved[at], np.float64(np.inf))
        control = exact(restarts[1]["un"], moved, masks["U"],
                        name="control.step1.u_m_s:un")
        require(not control["bit_identical"], "one-ULP control did not bind")
    first = next((row for row in rows if not row["bit_identical"]), None)
    return {
        "schema": "swg-dg-short-restart-ladder-v1",
        "label": "SWG_DG_SHORT_LADDER",
        "checkout_import": CHECKOUT_IMPORT,
        "legoesm_git_sha": legoesm_sha,
        "nemo_root": str(root.resolve()),
        "backend": backend,
        "gpu_device": gpu_device,
        "composition": composition,
        "precision_policy": (
            "fp64-native" if backend == "gpu" else "fp64-libm"),
        "rows": rows,
        "all_exact": first is None,
        "first_difference": first,
        "control": control,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--composition", choices=("outer-scan", "materialized"),
                        default="outer-scan")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-device")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--no-plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(not args.json.exists(), f"refusing to overwrite {args.json}")
        payload = run(
            args.nemo_root, composition=args.composition,
            backend=args.backend, gpu_device=args.gpu_device,
            plant=not args.no_plant)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - fail-closed diagnostic CLI
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    for row in payload["rows"]:
        print(
            f"step={row['step']} field={row['field']} "
            f"bit_unequal={row['bit_unequal']} max_abs={row['max_abs']:.17e}")
    first = payload["first_difference"]
    print(
        f"SWG_DG_SHORT_LADDER {'EXACT' if first is None else 'DEBT'} "
        f"composition={payload['composition']} backend={payload['backend']} "
        "first_difference=" + (
            "NONE" if first is None else
            f"step{first['step']}.{first['field']} bits={first['bit_unequal']}"))
    return 0 if payload["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
