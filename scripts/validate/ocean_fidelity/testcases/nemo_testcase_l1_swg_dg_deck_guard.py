#!/usr/bin/env python3
"""Compare NEMO's resolved SWG_DG namelist with legoESM's resolved card."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402

CHECKOUT_IMPORT = bootstrap_checkout(__file__)
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

ALLOWLIST = {
    "sn_cfctl.l_runstat": (
        "NEMO writes run.stat for completion evidence; legoESM has no NEMO "
        "run-stat writer, so this output-control switch cannot change state"),
    "nn_stock": (
        "NEMO writes annual restart tiles while legoESM writes a daily NPZ; "
        "restart cadence is output control and cannot change state"),
}

ALIASES = {
    "rn_dt": "rn_Dt_s",
    "rn_theta": "rn_theta_deg",
    "rn_domsiz": "rn_domsiz_m",
    "rn_dx": "rn_dx_m",
    "rn_tau": "rn_tau_pa",
    "rn_modified_grav": "rn_modified_grav_m_s2",
    "rn_rfr": "rn_rfr_s",
    "rn_beta": "rn_beta_m_s",
    "rn_f0": "rn_f0_s",
    "rn_fsp": "rn_fsp_s",
    "rn_time_dmp": "rn_time_dmp_days",
    "rn_time_dmp_out": "rn_time_dmp_out_days",
    "rn_rst_e3t": "rn_rst_e3t_days",
    "rn_lf_cutoff": "rn_lf_cutoff_days",
    "ln_usr": "ln_usr_sbc",
    "ln_dynadv_off": "ln_dynadv_OFF",
    "ln_dynvor_ent": "ln_dynvor_enT",
    "ln_dynldf_off": "ln_dynldf_OFF",
    "rn_uv": "rn_Uv_m_s",
    "rn_lv": "rn_Lv_m",
    "rn_ahm_b": "rn_ahm_b_m2_s",
}


def parse_resolved_echo(path: Path) -> dict[str, object]:
    """Parse scalar rows in NEMO's generated ``output.namelist.dyn``."""
    values = {}
    pattern = re.compile(r"\s*([A-Za-z0-9_%]+)\s*=\s*(.*?),\s*(?:\"|$)")
    for raw in path.read_text().splitlines():
        match = pattern.match(raw)
        if match:
            values[match.group(1).lower()] = _fortran_value(match.group(2))
    require(values, f"no resolved namelist values parsed from {path}")
    return values


def _fortran_value(raw: str) -> object:
    value = raw.strip()
    if value in {"T", "F"}:
        return value == "T"
    if value.startswith('"'):
        return value.strip('"').strip()
    repeated = re.fullmatch(r"(\d+)\*0", value)
    if repeated:
        return tuple(0 for _ in range(int(repeated.group(1))))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return value


def compare_resolved(nemo: dict[str, object], lego_namelist) -> dict:
    """Compare every common resolved field and allow only named I/O controls."""
    card = lego_namelist._asdict()
    controls = dict(card["sn_cfctl"])
    by_lower = {name.lower(): name for name in card}
    compared = []
    for nemo_name, nemo_value in nemo.items():
        if nemo_name.startswith("sn_cfctl%"):
            member = nemo_name.split("%", 1)[1]
            if member in controls:
                compared.append((f"sn_cfctl.{member}", nemo_value,
                                 controls[member]))
            continue
        candidate = ALIASES.get(nemo_name, nemo_name)
        card_name = by_lower.get(candidate.lower())
        if card_name is not None:
            compared.append((card_name, nemo_value, card[card_name]))
    require(len(compared) >= 130,
            f"resolved overlap unexpectedly small: {len(compared)}")
    differences = [
        {"field": name, "nemo": left, "legoesm": right,
         "allowlisted": name in ALLOWLIST,
         "reason": ALLOWLIST.get(name)}
        for name, left, right in compared if left != right
    ]
    refused = [row for row in differences if not row["allowlisted"]]
    require(not refused, f"non-allowlisted deck differences: {refused}")
    require(set(row["field"] for row in differences) == set(ALLOWLIST),
            "run-control allowlist did not bind exactly: "
            f"observed={[row['field'] for row in differences]}")
    return {
        "overlap_compared": len(compared),
        "differences": differences,
        "allowlist": ALLOWLIST,
        "all_physical_and_numerical_values_equal": True,
    }


def _load_driver():
    path = REPO_ROOT / "scripts/run/run_swg_double_gyre.py"
    spec = importlib.util.spec_from_file_location("swg_dg_deck_driver", path)
    require(spec is not None and spec.loader is not None,
            f"cannot load driver {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(nemo_namelist: Path, days: int) -> dict:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    driver = _load_driver()
    namelist = driver.resolved_daily_namelist(days)
    card = build_swg_card(namelist)
    comparison = compare_resolved(parse_resolved_echo(nemo_namelist), namelist)
    return {
        "schema": "swg-dg-cross-model-deck-guard-v1",
        "checkout_import": CHECKOUT_IMPORT,
        "nemo_resolved_namelist": str(nemo_namelist.resolve()),
        "legoesm_days": days,
        "legoesm_card_name": card.case,
        "legoesm_namelist": namelist._asdict(),
        **comparison,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-namelist", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.json.exists(), f"refusing to overwrite {args.json}")
        payload = run(args.nemo_namelist, args.days)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - fail-closed command boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        "SWG_DG_DECK_GUARD PASS "
        f"overlap={payload['overlap_compared']} "
        f"allowed_differences={len(payload['differences'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
