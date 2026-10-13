#!/usr/bin/env python3
"""Compare the preregistered year-5 GYRE r4_smag climate statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

GLOBAL_SHAPE = (82, 122)
SECONDS_PER_DAY = 86_400.0
FILTER_DAYS = 31
FILTER_LEAD_DAYS = FILTER_DAYS // 2
NEMO_MANIFEST = {
    "nemo": "3e7b30256344bdca1f9d99da9101adf94e24062e959416ae25f0852291a0b36f",
    "namelist_cfg": "97956dbbf895842f8728ba408167ac0875bef691215a851ecfc5fd6a73b5dbec",
    "namelist_ref": "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd",
}
PREREGISTRATION = {
    "claim": "trajectories diverge; only matched-window climate statistics are scored",
    "window_days": [1461, 1825],
    "highpass": "31-day uniform filter, nearest boundary, 15 preceding context days",
    "bars": {
        "window_mean_basin_ke_relative": 0.20,
        "eddy_fraction_relative": 0.20,
        "absolute_u_jet_row": 2.0,
        "ssh_zero_crossing_row": 2.0,
    },
    "report_only": [
        "time_mean_ssh", "time_mean_sst", "time_mean_speed",
        "highpass_eke_map", "kinetic_energy_wavenumber_spectrum",
    ],
}


class DailyFrame(NamedTuple):
    day: int
    ssh: np.ndarray
    u: np.ndarray
    v: np.ndarray
    sst: np.ndarray


class DailyProvider(NamedTuple):
    name: str
    sampling: str
    provenance: dict
    iterate: Callable[[], Iterator[DailyFrame]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_domain_tiles(
    tiles: list[tuple[np.ndarray, np.ndarray]], shape=GLOBAL_SHAPE,
) -> np.ndarray:
    """Place non-overlapping NEMO rank tiles using 1-based DOMAIN positions."""
    assembled = np.empty(shape, dtype=np.float64)
    filled = np.zeros(shape, dtype=bool)
    for first, raw in tiles:
        first = np.asarray(first, dtype=np.int64)
        values = np.asarray(raw, dtype=np.float64)
        require(first.shape == (2,) and values.ndim == 2, "invalid DOMAIN tile")
        i0, j0 = int(first[0] - 1), int(first[1] - 1)
        target = np.s_[j0:j0 + values.shape[0], i0:i0 + values.shape[1]]
        require(assembled[target].shape == values.shape, "DOMAIN tile exceeds global shape")
        require(not np.any(filled[target]), "overlapping DOMAIN tiles")
        require(np.all(np.isfinite(values)), "non-finite DOMAIN tile")
        assembled[target] = values
        filled[target] = True
    require(np.all(filled), f"DOMAIN assembly missed {np.count_nonzero(~filled)} cells")
    return assembled


def decode_weighted_surface_temperature(weighted: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Decode IOIPSL's mean(T*e3t) with the same-file mean(e3t)."""
    weighted = np.asarray(weighted, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    require(weighted.shape == thickness.shape, "weighted SST/thickness shapes differ")
    require(np.all(np.isfinite(weighted)) and np.all(thickness > 0.0),
            "invalid weighted SST or thickness")
    return weighted / thickness


def _read_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def _require_nemo_deck(root: Path) -> dict:
    manifest_path = root / "manifest.sha256"
    resolved = root / "resolved_namelist_lines.txt"
    require(manifest_path.is_file() and resolved.is_file(), "missing NEMO deck receipts")
    require(_read_manifest(manifest_path) == NEMO_MANIFEST, "NEMO manifest is not r4_smag")
    for name, digest in NEMO_MANIFEST.items():
        require((root / name).is_file() and sha256(root / name) == digest,
                f"NEMO deck digest mismatch: {name}")
    text = resolved.read_text(encoding="utf-8")
    for name, pattern in (
        ("nn_GYRE=4", r"\bnn_GYRE\s*=\s*4\b"),
        ("rn_Dt=3600", r"\brn_Dt\s*=\s*3600\."),
        ("ln_dynldf_lap=false", r"\bln_dynldf_lap\s*=\s*\.false\."),
        ("ln_dynldf_blp=true", r"\bln_dynldf_blp\s*=\s*\.true\."),
        ("nn_ahm_ijk_t=32", r"\bnn_ahm_ijk_t\s*=\s*32\b"),
        ("rn_csmc=3.5", r"\brn_csmc\s*=\s*3\.5\b"),
    ):
        require(re.search(pattern, text) is not None,
                f"resolved r4_smag deck missing {name}")
    return {
        "manifest_sha256": sha256(manifest_path),
        "resolved_namelist_sha256": sha256(resolved),
        "registered_files": NEMO_MANIFEST,
    }


def nemo_ioipsl_provider(root: Path, days: int, *, start_day: int) -> DailyProvider:
    """Read daily IOIPSL means; SST is mean(T*e3t)/mean(e3t)."""
    import netCDF4

    require(start_day > 0 and days > 0, "NEMO window must be positive")
    deck = _require_nemo_deck(root)
    paths = {kind: sorted(root.glob(f"*grid_{kind}_*.nc")) for kind in "TUV"}
    require({kind: len(value) for kind, value in paths.items()} == {"T": 8, "U": 8, "V": 8},
            "NEMO r4_smag requires eight tiles for each T/U/V field")
    stop = start_day + days - 1

    def iterate() -> Iterator[DailyFrame]:
        with ExitStack() as stack:
            datasets = {
                kind: [stack.enter_context(netCDF4.Dataset(path)) for path in found]
                for kind, found in paths.items()
            }
            for kind, dsets in datasets.items():
                variable = {"T": "sossheig", "U": "vozocrtx", "V": "vomecrty"}[kind]
                for ds in dsets:
                    ds.set_auto_mask(False)
                    require(np.array_equal(ds.DOMAIN_size_global, [122, 82]),
                            "NEMO DOMAIN_size_global is not 122x82")
                    require(getattr(ds[variable], "online_operation", None) == "ave(x)",
                            f"{variable} is not an IOIPSL daily mean")
                    require(float(ds[variable].interval_operation) == 3600.0
                            and float(ds[variable].interval_write) == SECONDS_PER_DAY,
                            f"{variable} cadence differs")
            for day in range(start_day, stop + 1):
                index = day - 1
                t_tiles, sst_tiles, u_tiles, v_tiles = [], [], [], []
                for ds in datasets["T"]:
                    first = ds.DOMAIN_position_first
                    ssh = np.asarray(ds["sossheig"][index])
                    weighted_t = np.asarray(ds["sosstsst"][index])
                    thickness = np.asarray(ds["vovvle3t"][index, 0])
                    t_tiles.append((first, ssh))
                    sst_tiles.append((first, decode_weighted_surface_temperature(
                        weighted_t, thickness)))
                for kind, variable, tiles in (
                    ("U", "vozocrtx", u_tiles), ("V", "vomecrty", v_tiles)
                ):
                    for ds in datasets[kind]:
                        tiles.append((ds.DOMAIN_position_first,
                                      np.asarray(ds[variable][index, 0])))
                yield DailyFrame(
                    day,
                    assemble_domain_tiles(t_tiles),
                    assemble_domain_tiles(u_tiles),
                    assemble_domain_tiles(v_tiles),
                    assemble_domain_tiles(sst_tiles),
                )

    files = {kind: [path.name for path in found] for kind, found in paths.items()}
    return DailyProvider(
        "NEMO", "daily_mean",
        {"root": str(root.resolve()), "window_days": [start_day, stop],
         "tiles": files, "deck": deck, "sst_convention": "sosstsst/vovvle3t same file"},
        iterate,
    )


def _require_legoesm_deck(namelist: dict) -> None:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import GYRE_R4_SMAG_NAMELIST

    expected = json.loads(json.dumps(GYRE_R4_SMAG_NAMELIST._asdict()))
    for field in expected.keys() - {"nn_itend", "nn_stock", "nn_write"}:
        require(namelist.get(field) == expected[field],
                f"legoESM manifest differs from r4_smag at {field}")


def legoesm_npz_provider(root: Path, days: int, *, start_day: int) -> DailyProvider:
    require(start_day > 0 and days > 0, "legoESM window must be positive")
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing complete manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stop = start_day + days - 1
    require(manifest.get("schema") == "legoesm-gyre-hires-daily-v1"
            and manifest.get("status") == "complete"
            and manifest.get("completed_days") == manifest.get("days")
            and manifest.get("days", 0) >= stop, "incomplete legoESM climate run")
    require(manifest.get("sampling") == ["end_of_day_snapshot", "daily_mean"],
            "legoESM run does not contain daily means")
    _require_legoesm_deck(manifest.get("namelist", {}))
    pins = manifest.get("daily_sha256", {})
    fields = (
        "ssh_m_daily_mean", "surface_u_m_s_daily_mean",
        "surface_v_m_s_daily_mean", "surface_temperature_c_daily_mean",
    )
    expected = [root / f"day{day:04d}.npz" for day in range(start_day, stop + 1)]
    for path in expected:
        require(path.is_file() and pins.get(path.name) == sha256(path),
                f"legoESM daily digest mismatch: {path.name}")

    def iterate() -> Iterator[DailyFrame]:
        for day, path in zip(range(start_day, stop + 1), expected, strict=True):
            with np.load(path, allow_pickle=False) as data:
                require(int(data["day"]) == day and float(data["dt_s"]) == 3600.0,
                        f"{path.name}: embedded clock differs")
                values = [np.asarray(data[name], dtype=np.float64) for name in fields]
            require(all(value.shape == GLOBAL_SHAPE for value in values),
                    f"{path.name}: field shape differs")
            yield DailyFrame(day, *values)

    return DailyProvider(
        "legoESM", "daily_mean",
        {"root": str(root.resolve()), "window_days": [start_day, stop],
         "manifest_sha256": sha256(manifest_path),
         "legoesm_git_sha": manifest.get("legoesm_git_sha"),
         "mode": manifest.get("mode"), "certified_policy": manifest.get("certified_policy"),
         "sst_convention": manifest.get("daily_mean_convention", {}).get("surface_temperature_c")},
        iterate,
    )


def geometry() -> dict:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_gyre_zco_card,
        gyre_horizontal_coordinates,
    )

    card = build_gyre_zco_card(GYRE_R4_SMAG_NAMELIST)
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active)[..., 0] & wet
    u = active & np.roll(active, -1, axis=1)
    v = active & np.roll(active, -1, axis=0)
    u[:, -1] = False
    v[-1] = False
    coords = gyre_horizontal_coordinates(GYRE_R4_SMAG_NAMELIST)
    return {
        "wet": wet, "u_mask": u, "v_mask": v,
        "area": np.asarray(card.recipe.grid.area),
        "latitude": np.asarray(coords["gphit"]),
        "f": np.asarray(coords["ff_t"]),
        "spacing_m": float(np.asarray(card.recipe.grid.dx_T)[0, 0]),
    }


def cell_velocity(
    u_native: np.ndarray, v_native: np.ndarray, geom: dict,
) -> tuple[np.ndarray, np.ndarray]:
    from legoesm.ocean.diagnostics import velocity_to_cell_centre

    u_native = np.where(geom["u_mask"], u_native, 0.0)
    v_native = np.where(geom["v_mask"], v_native, 0.0)
    u = np.concatenate([np.zeros_like(u_native[:, :1]), u_native], axis=1)
    v = np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)
    u_c, v_c = velocity_to_cell_centre(u, v)
    return np.asarray(u_c), np.asarray(v_c)


def _ssh_zero_crossing(mean_ssh: np.ndarray, geom: dict, near_row: float) -> tuple[float, float]:
    wet = geom["wet"]
    profile = np.array([
        np.mean(mean_ssh[j, wet[j]], dtype=np.float64) if np.any(wet[j]) else np.nan
        for j in range(wet.shape[0])
    ])
    candidates = []
    for j in range(profile.size - 1):
        if np.isfinite(profile[j:j + 2]).all() and profile[j] * profile[j + 1] <= 0.0:
            fraction = (
                0.0 if profile[j] == profile[j + 1]
                else -profile[j] / (profile[j + 1] - profile[j])
            )
            candidates.append(j + fraction)
    require(bool(candidates), "mean SSH has no zonal-mean zero crossing")
    row = min(candidates, key=lambda value: abs(value - near_row))
    row_lat = np.array([
        np.mean(geom["latitude"][j, wet[j]]) if np.any(wet[j]) else np.nan
        for j in range(wet.shape[0])
    ])
    lo = min(int(np.floor(row)), row_lat.size - 2)
    latitude = row_lat[lo] + (row - lo) * (row_lat[lo + 1] - row_lat[lo])
    return float(row), float(latitude)


def diagnose(provider: DailyProvider, *, target_start: int, target_days: int, geom=None) -> dict:
    from legoesm.ocean.diagnostics import isotropic_energy_spectrum
    from scipy.ndimage import uniform_filter1d

    geom = geometry() if geom is None else geom
    raw = list(provider.iterate())
    require(bool(raw), f"{provider.name}: empty window")
    lead = target_start - raw[0].day
    require(0 <= lead and lead + target_days <= len(raw), "target outside provider window")
    require([frame.day for frame in raw] == list(range(raw[0].day, raw[0].day + len(raw))),
            f"{provider.name}: non-contiguous days")
    ssh = np.stack([frame.ssh for frame in raw])
    sst = np.stack([frame.sst for frame in raw])
    native_u = np.stack([frame.u for frame in raw])
    velocities = [cell_velocity(frame.u, frame.v, geom) for frame in raw]
    u = np.stack([value[0] for value in velocities])
    v = np.stack([value[1] for value in velocities])
    target = np.s_[lead:lead + target_days]
    wet = geom["wet"]
    area = np.where(wet, geom["area"], 0.0)
    area_sum = np.sum(area, dtype=np.float64)
    speed2 = u[target] ** 2 + v[target] ** 2
    ke_series = np.sum(0.5 * speed2 * area, axis=(1, 2), dtype=np.float64) / area_sum
    u_hp = u - uniform_filter1d(u, FILTER_DAYS, axis=0, mode="nearest")
    v_hp = v - uniform_filter1d(v, FILTER_DAYS, axis=0, mode="nearest")
    eke_map = np.mean(0.5 * (u_hp[target] ** 2 + v_hp[target] ** 2), axis=0)
    basin_eke = float(np.sum(eke_map * area, dtype=np.float64) / area_sum)
    mean_ke = float(np.mean(ke_series, dtype=np.float64))
    mean_abs_u_by_row = np.mean(np.abs(native_u[target]), axis=(0, 2))
    valid_rows = np.any(geom["u_mask"], axis=1)
    absolute_u_jet_row = int(np.flatnonzero(valid_rows)[np.argmax(mean_abs_u_by_row[valid_rows])])
    mean_ssh = np.mean(ssh[target], axis=0)
    ssh_row, ssh_lat = _ssh_zero_crossing(mean_ssh, geom, absolute_u_jet_row)
    wet_indices = np.argwhere(wet)
    j0, i0 = wet_indices.min(axis=0)
    j1, i1 = wet_indices.max(axis=0) + 1
    require(np.all(wet[j0:j1, i0:i1]), "spectral wet basin is not rectangular")
    spectrum = None
    wavenumber = None
    for uu, vv in zip(u[target], v[target], strict=True):
        k, e = isotropic_energy_spectrum(
            uu[j0:j1, i0:i1], vv[j0:j1, i0:i1], geom["spacing_m"]
        )
        k, e = np.asarray(k), np.asarray(e)
        if spectrum is None:
            wavenumber, spectrum = k, np.zeros_like(e)
        spectrum += e
    return {
        "basin_mean_ke_series_m2_s2": ke_series,
        "window_mean_basin_ke_m2_s2": np.asarray(mean_ke),
        "highpass_eke_map_m2_s2": eke_map,
        "basin_mean_highpass_eke_m2_s2": np.asarray(basin_eke),
        "eddy_fraction": np.asarray(basin_eke / mean_ke),
        "absolute_u_jet_row": np.asarray(absolute_u_jet_row),
        "ssh_zero_crossing_row": np.asarray(ssh_row),
        "ssh_zero_crossing_latitude_deg": np.asarray(ssh_lat),
        "time_mean_ssh_m": mean_ssh,
        "time_mean_sst_c": np.mean(sst[target], axis=0),
        "time_mean_speed_m_s": np.mean(np.sqrt(speed2), axis=0),
        "spectrum_wavenumber_m-1": wavenumber,
        "mean_ke_spectrum_m3_s2": spectrum / target_days,
    }


def score_statistics(results: dict[str, dict]) -> dict:
    nemo, lego = results["NEMO"], results["legoESM"]
    rows = []
    for name, bar, mode in (
        ("window_mean_basin_ke_m2_s2", 0.20, "relative"),
        ("eddy_fraction", 0.20, "relative"),
        ("absolute_u_jet_row", 2.0, "absolute_rows"),
        ("ssh_zero_crossing_row", 2.0, "absolute_rows"),
    ):
        reference, candidate = float(nemo[name]), float(lego[name])
        error = (
            abs(candidate - reference) / abs(reference)
            if mode == "relative" else abs(candidate - reference)
        )
        rows.append({"name": name, "NEMO": reference, "legoESM": candidate,
                     "error": error, "bar": bar, "mode": mode,
                     "status": "AT-BAR" if error <= bar else "DEBT"})
    passed = all(row["status"] == "AT-BAR" for row in rows)
    return {"rows": rows, "verdict": "PASS" if passed else "DEBT"}


def _array_summary(value) -> dict:
    array = np.asarray(value)
    return {"shape": list(array.shape), "min": float(np.min(array)),
            "max": float(np.max(array)), "mean": float(np.mean(array, dtype=np.float64)),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def write_outputs(
    output: Path, results: dict, providers: tuple[DailyProvider, DailyProvider],
    *, start_day: int, days: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "gyre_hires_climate_arrays.npz"
    np.savez_compressed(arrays_path, **{
        f"{model}.{name}": value for model, metrics in results.items()
        for name, value in metrics.items()
    })
    score = score_statistics(results)
    report = {
        "schema": "gyre-r4-smag-climate-comparison-v1",
        "preregistration": PREREGISTRATION,
        "window_days": [start_day, start_day + days - 1],
        "sampling": "daily_mean",
        "provenance": {provider.name: provider.provenance for provider in providers},
        "metrics": {model: {name: _array_summary(value) for name, value in metrics.items()}
                    for model, metrics in results.items()},
        "acceptance": score,
        "artifacts": {arrays_path.name: sha256(arrays_path)},
    }
    path = output / "gyre_hires_climate_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--legoesm-root", type=Path, required=True)
    parser.add_argument("--start-day", type=int, default=1461)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        context_start = max(1, args.start_day - FILTER_LEAD_DAYS)
        context_days = args.start_day + args.days - context_start
        providers = (
            nemo_ioipsl_provider(args.nemo_root, context_days, start_day=context_start),
            legoesm_npz_provider(args.legoesm_root, context_days, start_day=context_start),
        )
        require(providers[0].sampling == providers[1].sampling == "daily_mean",
                "model sampling conventions differ")
        geom = geometry()
        results = {provider.name: diagnose(provider, target_start=args.start_day,
                                            target_days=args.days, geom=geom)
                   for provider in providers}
        report = write_outputs(args.output, results, providers,
                               start_day=args.start_day, days=args.days)
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"{report['acceptance']['verdict']} -> {args.output}")
    return 0 if report["acceptance"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
