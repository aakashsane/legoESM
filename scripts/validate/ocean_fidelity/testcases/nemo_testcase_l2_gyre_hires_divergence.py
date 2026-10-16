#!/usr/bin/env python3
"""Measure GYRE twin divergence and yearly eddy fractions on one protocol.

The divergence definition is the operator's round-8 definition: RMS of the
difference divided by the RMS spatial anomaly of the certified vector-math
NEMO field, plus pattern correlation, on daily-mean SSH and native surface U.
The eddy fraction is the climate scorer's 31-day nearest-boundary high-pass
EKE divided by surface KE, evaluated from the full five-year series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from nemo_testcase_l2_gyre_hires_climate import (  # noqa: E402
    FILTER_DAYS,
    GLOBAL_SHAPE,
    DailyProvider,
    assemble_domain_tiles,
    geometry,
    legoesm_npz_provider,
    nemo_ioipsl_provider,
    sha256,
)
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

TOTAL_DAYS = 1825
CURVE_DAYS = 365
CHECKPOINT_DAYS = (1, 10, 30, 60, 90, 120, 180, 240, 300, 365)
# Independent operator probe, printed to two digits after the decimal in
# scientific notation.  String equality preserves that probe's real precision
# instead of inventing a tighter numeric tolerance after the fact.
OPERATOR_TWIN_SSH_BENCHMARK = {
    30: "1.04e-09",
    60: "1.17e-09",
    90: "1.19e-08",
    120: "1.21e-06",
    180: "2.95e-04",
    365: "3.89e-03",
}


def normalized_rms_and_correlation(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    """Score two same-stagger daily fields on a common active mask."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(reference.shape == candidate.shape == mask.shape,
            "divergence field/mask shapes differ")
    require(np.any(mask), "divergence mask is empty")
    left, right = reference[mask], candidate[mask]
    require(np.all(np.isfinite(left)) and np.all(np.isfinite(right)),
            "non-finite active divergence field")
    left_anomaly = left - np.mean(left, dtype=np.float64)
    right_anomaly = right - np.mean(right, dtype=np.float64)
    reference_rms = float(np.sqrt(np.mean(
        left_anomaly * left_anomaly, dtype=np.float64)))
    require(reference_rms > 0.0, "reference spatial anomaly is zero")
    denominator = np.sqrt(
        np.sum(left_anomaly * left_anomaly, dtype=np.float64)
        * np.sum(right_anomaly * right_anomaly, dtype=np.float64)
    )
    require(denominator > 0.0, "pattern-correlation denominator is zero")
    return {
        "normalized_rms_difference": float(np.sqrt(np.mean(
            (left - right) ** 2, dtype=np.float64)) / reference_rms),
        "pattern_correlation": float(np.sum(
            left_anomaly * right_anomaly, dtype=np.float64) / denominator),
    }


def divergence_curve(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the registered pointwise score to every matched day."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    require(reference.shape == candidate.shape and reference.ndim == 3,
            "divergence time series shapes differ")
    rows = [
        normalized_rms_and_correlation(left, right, mask)
        for left, right in zip(reference, candidate, strict=True)
    ]
    return {
        "normalized_rms_difference": np.asarray([
            row["normalized_rms_difference"] for row in rows]),
        "pattern_correlation": np.asarray([
            row["pattern_correlation"] for row in rows]),
    }


def yearly_eddy_statistics(
    u: np.ndarray, v: np.ndarray, wet: np.ndarray,
) -> list[dict[str, float | int]]:
    """Return full-series yearly KE/EKE using the registered 31-day filter."""
    from scipy.ndimage import uniform_filter1d

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    require(u.shape == v.shape and u.ndim == 3 and u.shape[1:] == wet.shape,
            "eddy series/mask shapes differ")
    require(u.shape[0] % 365 == 0 and np.any(wet),
            "eddy series is not an integer number of non-empty years")
    require(np.all(np.isfinite(u[:, wet])) and np.all(np.isfinite(v[:, wet])),
            "non-finite active eddy series")
    u_hp = u - uniform_filter1d(u, FILTER_DAYS, axis=0, mode="nearest")
    v_hp = v - uniform_filter1d(v, FILTER_DAYS, axis=0, mode="nearest")
    ke = 0.5 * (u * u + v * v)
    eke = 0.5 * (u_hp * u_hp + v_hp * v_hp)
    result = []
    for year in range(u.shape[0] // 365):
        window = np.s_[365 * year:365 * (year + 1)]
        mean_ke = float(np.mean(ke[window][:, wet], dtype=np.float64))
        mean_eke = float(np.mean(eke[window][:, wet], dtype=np.float64))
        require(mean_ke > 0.0, "yearly surface KE is zero")
        result.append({
            "year": year + 1,
            "surface_ke_m2_s2": mean_ke,
            "highpass_eke_m2_s2": mean_eke,
            "eddy_fraction": mean_eke / mean_ke,
        })
    return result


def nemo_surface_masks(root: Path) -> dict[str, np.ndarray]:
    """Stitch NEMO surface T/U masks from their own DOMAIN metadata."""
    import netCDF4

    paths = sorted(root.glob("mesh_mask_????.nc"))
    require(len(paths) == 8, "NEMO reference requires eight mesh-mask tiles")
    tiles = {"wet": [], "u": []}
    for path in paths:
        with netCDF4.Dataset(path) as dataset:
            require(np.array_equal(dataset.DOMAIN_size_global, [122, 82]),
                    "mesh-mask DOMAIN_size_global is not 122x82")
            first = np.asarray(dataset.DOMAIN_position_first)
            tiles["wet"].append((first, np.asarray(dataset["tmask"][0, 0])))
            tiles["u"].append((first, np.asarray(dataset["umask"][0, 0])))
    return {
        name: assemble_domain_tiles(value) > 0.5
        for name, value in tiles.items()
    }


def load_surface_series(
    provider: DailyProvider,
    *,
    total_days: int = TOTAL_DAYS,
    curve_days: int = CURVE_DAYS,
) -> dict:
    """Stream one provider once; retain only fields used by this instrument."""
    require(0 < curve_days <= total_days, "invalid divergence window")
    ssh = np.empty((curve_days, *GLOBAL_SHAPE), dtype=np.float64)
    u = np.empty((total_days, *GLOBAL_SHAPE), dtype=np.float64)
    v = np.empty_like(u)
    count = 0
    for count, frame in enumerate(provider.iterate(), start=1):
        require(count <= total_days and frame.day == count,
                f"{provider.name}: daily sequence differs")
        if count <= curve_days:
            ssh[count - 1] = frame.ssh
        u[count - 1] = frame.u
        v[count - 1] = frame.v
    require(count == total_days, f"{provider.name}: expected {total_days} days")
    return {"ssh": ssh, "u": u, "v": v, "provenance": provider.provenance}


def _checkpoint_rows(curves: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for day in CHECKPOINT_DAYS:
        row = {"day": day}
        for comparison, fields in curves.items():
            for field, values in fields.items():
                row[f"{comparison}.{field}.normalized_rms_difference"] = float(
                    values["normalized_rms_difference"][day - 1])
                row[f"{comparison}.{field}.pattern_correlation"] = float(
                    values["pattern_correlation"][day - 1])
        rows.append(row)
    return rows


def _validate_twin(curve: dict[str, dict[str, np.ndarray]]) -> dict:
    measured = {
        day: f"{curve['ssh']['normalized_rms_difference'][day - 1]:.2e}"
        for day in OPERATOR_TWIN_SSH_BENCHMARK
    }
    require(measured == OPERATOR_TWIN_SSH_BENCHMARK,
            "NEMO twin curve does not reproduce the independent benchmark")
    return {
        "status": "PASS",
        "quantity": "SSH normalized RMS difference",
        "precision": "two digits after decimal in scientific notation",
        "expected": OPERATOR_TWIN_SSH_BENCHMARK,
        "measured": measured,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-nemo-root", type=Path, required=True)
    parser.add_argument("--scalar-nemo-root", type=Path, required=True)
    parser.add_argument("--scalar-nemo-manifest", type=Path, required=True)
    parser.add_argument("--scalar-nemo-label", required=True)
    parser.add_argument("--baseline-legoesm-run-label", required=True)
    parser.add_argument("--candidate-legoesm-run-label", required=True)
    parser.add_argument(
        "--run-registry", type=Path,
        default=HERE / "gyre_hires_registered_runs.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        vector_provider = nemo_ioipsl_provider(
            args.vector_nemo_root, TOTAL_DAYS, start_day=1)
        scalar_provider = nemo_ioipsl_provider(
            args.scalar_nemo_root,
            TOTAL_DAYS,
            start_day=1,
            alternative_manifest=args.scalar_nemo_manifest,
            alternative_label=args.scalar_nemo_label,
        )
        baseline_provider = legoesm_npz_provider(
            args.baseline_legoesm_run_label,
            TOTAL_DAYS,
            start_day=1,
            registry_path=args.run_registry,
        )
        candidate_provider = legoesm_npz_provider(
            args.candidate_legoesm_run_label,
            TOTAL_DAYS,
            start_day=1,
            registry_path=args.run_registry,
        )

        masks = nemo_surface_masks(args.vector_nemo_root)
        card_geom = geometry()
        require(np.array_equal(masks["wet"], card_geom["wet"]),
                "NEMO and legoESM T masks differ")
        require(np.array_equal(masks["u"], card_geom["u_mask"]),
                "NEMO and legoESM U masks differ")

        vector = load_surface_series(vector_provider)
        vector_years = yearly_eddy_statistics(
            vector["u"], vector["v"], masks["wet"])
        vector_curve = {"ssh": vector["ssh"], "u": vector["u"][:CURVE_DAYS]}
        del vector["v"]

        curves: dict[str, dict[str, dict]] = {}
        yearly = {"nemo_vector": vector_years}
        provenance = {"nemo_vector": vector["provenance"]}
        providers = (
            ("nemo_scalar", scalar_provider),
            ("legoesm_baseline", baseline_provider),
            ("legoesm_candidate", candidate_provider),
        )
        for label, provider in providers:
            series = load_surface_series(provider)
            yearly[label] = yearly_eddy_statistics(
                series["u"], series["v"], masks["wet"])
            curves[label] = {
                field: divergence_curve(vector_curve[field], series[field][
                    :CURVE_DAYS] if field == "u" else series[field], mask)
                for field, mask in (("ssh", masks["wet"]), ("u", masks["u"]))
            }
            provenance[label] = series["provenance"]
            del series

        twin_validation = _validate_twin(curves["nemo_scalar"])
        exact_control = normalized_rms_and_correlation(
            vector_curve["ssh"][0], vector_curve["ssh"][0], masks["wet"])
        planted = np.array(vector_curve["ssh"][0], copy=True)
        plant_index = tuple(np.argwhere(masks["wet"])[0])
        planted[plant_index] = np.nextafter(planted[plant_index], np.inf)
        plant_control = normalized_rms_and_correlation(
            vector_curve["ssh"][0], planted, masks["wet"])
        require(exact_control["normalized_rms_difference"] == 0.0
                and plant_control["normalized_rms_difference"] > 0.0,
                "divergence exact/one-ULP controls did not separate")

        arrays_path = args.output / "gyre_hires_divergence_arrays.npz"
        np.savez_compressed(arrays_path, **{
            f"{comparison}.{field}.{metric}": value
            for comparison, fields in curves.items()
            for field, metrics in fields.items()
            for metric, value in metrics.items()
        })
        report = {
            "schema": "gyre-r4-smag-divergence-envelope-v1",
            "definition": {
                "divergence": (
                    "RMS(candidate-reference) / RMS(reference spatial anomaly)"
                ),
                "correlation": "same-day spatial anomaly pattern correlation",
                "reference": "certified vector-math NEMO daily mean",
                "fields": ["SSH on T mask", "native surface U on U mask"],
                "curve_days": [1, CURVE_DAYS],
                "eddy_fraction": (
                    "31-day nearest-boundary high-pass native U/V EKE / KE"
                ),
                "eddy_years": [1, TOTAL_DAYS // 365],
            },
            "provenance": provenance,
            "mask_identity": {
                "T": "NEMO mesh mask equals legoESM card mask",
                "U": "NEMO mesh mask equals legoESM card mask",
            },
            "instrument_validation": {
                "nemo_twin_benchmark": twin_validation,
                "exact_control": exact_control,
                "one_ulp_control": {
                    **plant_control,
                    "index": [int(value) for value in plant_index],
                    "status": "FIRED",
                },
            },
            "checkpoints": _checkpoint_rows(curves),
            "yearly_statistics": yearly,
            "artifacts": {arrays_path.name: sha256(arrays_path)},
        }
        report_path = args.output / "gyre_hires_divergence_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
