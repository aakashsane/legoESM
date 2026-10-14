#!/usr/bin/env python3
"""Compare every NEMO compiled-read SWG_DG value with legoESM's card."""
from __future__ import annotations

import argparse
import hashlib
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
from nemo_testcase_l1_swg_namelist_inventory import (  # noqa: E402
    EXPECTED_CARD_READS,
)
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


def _without_fortran_comment(raw: str) -> str:
    """Remove a Fortran comment while retaining exclamation marks in strings."""
    quote = None
    for index, character in enumerate(raw):
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif character == "!" and quote is None:
            return raw[:index]
    return raw


def parse_namelist_groups(path: Path) -> dict[str, dict[str, object]]:
    """Parse scalar assignments by group from a NEMO namelist or echo."""
    require(path.is_file(), f"missing NEMO namelist source {path}")
    groups: dict[str, dict[str, object]] = {}
    group = None
    assignment = re.compile(
        r"([A-Za-z][A-Za-z0-9_%]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*,?$")
    for raw in path.read_text().splitlines():
        line = _without_fortran_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            group = line[1:].split()[0].lower()
            groups.setdefault(group, {})
            continue
        if line == "/":
            group = None
            continue
        if group is None:
            continue
        match = assignment.fullmatch(line)
        if match:
            groups[group][match.group(1).lower()] = _fortran_value(
                match.group(2))
    require(groups, f"no namelist groups parsed from {path}")
    return groups


def parse_resolved_echo(path: Path) -> dict[str, object]:
    """Return the legacy flat view of NEMO's generated resolved echo."""
    return {
        name: value
        for fields in parse_namelist_groups(path).values()
        for name, value in fields.items()
    }


def _fortran_value(raw: str) -> object:
    value = raw.strip().rstrip(",").strip()
    lower = value.lower()
    if lower in {"t", ".true."}:
        return True
    if lower in {"f", ".false."}:
        return False
    if ((value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))):
        return value[1:-1].strip()
    repeated = re.fullmatch(r"(\d+)\s*\*\s*([+-]?[0-9]+)", value)
    if repeated:
        return tuple(
            int(repeated.group(2)) for _ in range(int(repeated.group(1))))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run_input_hashes(run_dir: Path) -> dict[str, str]:
    """Require the two namelists to match the run's pre-execution manifest."""
    manifest = run_dir / "manifest.sha256"
    require(manifest.is_file(), f"missing run manifest {manifest}")
    recorded = {}
    for raw in manifest.read_text().splitlines():
        parts = raw.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            recorded[parts[1].lstrip("*")] = parts[0]
    result = {}
    for name in ("namelist_ref", "namelist_cfg"):
        path = run_dir / name
        require(name in recorded, f"run manifest does not hash {name}")
        actual = _sha256(path)
        require(actual == recorded[name],
                f"run input hash differs for {name}: "
                f"recorded={recorded[name]} actual={actual}")
        result[name] = actual
    return result


def _nemo_spelling(card_field: str) -> str:
    inverse = {card.lower(): nemo for nemo, card in ALIASES.items()}
    return inverse.get(card_field.lower(), card_field.lower())


def _resolve_one(group: str, raw_name: str,
                 echo: dict[str, dict[str, object]],
                 reference: dict[str, dict[str, object]],
                 config: dict[str, dict[str, object]]) -> tuple[object, str]:
    """Resolve one value using echo first, then NEMO ref->cfg precedence."""
    if raw_name in echo.get(group, {}):
        return echo[group][raw_name], "output.namelist.dyn"
    if raw_name in config.get(group, {}):
        return config[group][raw_name], "namelist_cfg"
    if raw_name in reference.get(group, {}):
        return reference[group][raw_name], "namelist_ref"
    raise RuntimeError(
        f"unresolved compiled-read field {group}.{raw_name}: absent from "
        "output.namelist.dyn, namelist_cfg, and namelist_ref")


def _comparison_row(*, group: str, inventory_field: str, field: str,
                    nemo_value: object, lego_value: object,
                    source: str) -> dict:
    equal = nemo_value == lego_value
    return {
        "group": group,
        "inventory_field": inventory_field,
        "field": field,
        "nemo": nemo_value,
        "legoesm": lego_value,
        "source": source,
        "equal": equal,
        "allowlisted": not equal and field in ALLOWLIST,
        "reason": ALLOWLIST.get(field) if not equal else None,
    }


def resolve_compiled_inventory(
        echo: dict[str, dict[str, object]],
        reference: dict[str, dict[str, object]],
        config: dict[str, dict[str, object]], lego_namelist,
) -> tuple[list[dict], dict[str, str]]:
    """Resolve all 131 compiled-read fields and compare their full values."""
    card = lego_namelist._asdict()
    card_by_lower = {name.lower(): name for name in card}
    rows = []
    sources = {}
    resolved_fields = 0
    for group, fields in EXPECTED_CARD_READS.items():
        for inventory_field in fields:
            if inventory_field == "sn_cfctl":
                controls = dict(card[inventory_field])
                for member, lego_value in controls.items():
                    raw_name = f"sn_cfctl%{member}".lower()
                    nemo_value, source = _resolve_one(
                        group, raw_name, echo, reference, config)
                    rows.append(_comparison_row(
                        group=group, inventory_field=inventory_field,
                        field=f"sn_cfctl.{member}", nemo_value=nemo_value,
                        lego_value=lego_value, source=source))
                    sources[f"{group}.{raw_name}"] = source
                resolved_fields += 1
                continue
            card_name = card_by_lower.get(inventory_field.lower())
            require(card_name is not None,
                    f"compiled-read field missing from legoESM card: "
                    f"{group}.{inventory_field}")
            raw_name = _nemo_spelling(inventory_field)
            nemo_value, source = _resolve_one(
                group, raw_name, echo, reference, config)
            rows.append(_comparison_row(
                group=group, inventory_field=inventory_field,
                field=card_name, nemo_value=nemo_value,
                lego_value=card[card_name], source=source))
            sources[f"{group}.{raw_name}"] = source
            resolved_fields += 1
    require(resolved_fields == 131,
            f"compiled-read resolution is not exhaustive: {resolved_fields}/131")
    return rows, sources


def compare_resolved(
        echo: dict[str, dict[str, object]],
        reference: dict[str, dict[str, object]],
        config: dict[str, dict[str, object]], lego_namelist,
) -> dict:
    """Compare all compiled reads plus common case-specific echo values."""
    rows, sources = resolve_compiled_inventory(
        echo, reference, config, lego_namelist)
    card = lego_namelist._asdict()
    card_by_lower = {name.lower(): name for name in card}
    inventory = {
        field.lower()
        for fields in EXPECTED_CARD_READS.values()
        for field in fields
    }
    extras = []
    for group, values in echo.items():
        for raw_name, nemo_value in values.items():
            if raw_name.startswith("sn_cfctl%"):
                continue
            candidate = ALIASES.get(raw_name, raw_name)
            card_name = card_by_lower.get(candidate.lower())
            if card_name is None or card_name.lower() in inventory:
                continue
            extras.append(_comparison_row(
                group=group, inventory_field="case-specific-echo",
                field=card_name, nemo_value=nemo_value,
                lego_value=card[card_name], source="output.namelist.dyn"))
    rows.extend(extras)
    differences = [row for row in rows if not row["equal"]]
    refused = [row for row in differences if not row["allowlisted"]]
    require(not refused, f"non-allowlisted deck differences: {refused}")
    require({row["field"] for row in differences} == set(ALLOWLIST),
            "run-control allowlist did not bind exactly: "
            f"observed={[row['field'] for row in differences]}")
    all_non_allowlisted_equal = not refused
    return {
        "compiled_inventory_fields_resolved": 131,
        "compiled_inventory_fields_required": 131,
        "comparison_rows": len(rows),
        "case_specific_echo_rows": len(extras),
        "resolved_sources": sources,
        "rows": rows,
        "differences": differences,
        "allowlist": ALLOWLIST,
        "all_physical_and_numerical_values_equal": all_non_allowlisted_equal,
        "all_non_allowlisted_values_equal": all_non_allowlisted_equal,
    }


def crosscheck_fallback_groups(
        ocean_output: Path, comparison: dict,
) -> dict[str, dict]:
    """Check fallback-resolved values when ocean.output prints them."""
    require(ocean_output.is_file(), f"missing NEMO output {ocean_output}")
    text = ocean_output.read_text(errors="replace")
    by_group = {}
    for row in comparison["rows"]:
        if row["inventory_field"] == "case-specific-echo":
            continue
        if row["source"] == "output.namelist.dyn":
            continue
        raw_name = _nemo_spelling(row["inventory_field"])
        if row["inventory_field"] == "sn_cfctl":
            raw_name = row["field"].replace(".", "%")
        pattern = re.compile(
            rf"\b{re.escape(raw_name)}\s*=\s*([^\s,]+)", re.IGNORECASE)
        found = pattern.findall(text)
        group_result = by_group.setdefault(row["group"], {
            "status": "not_printed",
            "matched_fields": [],
            "note": (
                "NEMO's ocean.output contains no machine-readable value for "
                "this echo-omitted group; the content-hashed namelist_ref/"
                "namelist_cfg precedence is authoritative"),
        })
        if not found:
            continue
        printed = _fortran_value(found[-1])
        require(printed == row["nemo"],
                f"ocean.output cross-check differs for {row['group']}."
                f"{raw_name}: printed={printed!r} resolved={row['nemo']!r}")
        group_result["status"] = "matched"
        group_result["matched_fields"].append(raw_name)
        group_result["note"] = "printed value(s) agree with resolved inputs"
    return by_group


def _load_driver():
    path = REPO_ROOT / "scripts/run/run_swg_double_gyre.py"
    spec = importlib.util.spec_from_file_location("swg_dg_deck_driver", path)
    require(spec is not None and spec.loader is not None,
            f"cannot load driver {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(nemo_run_dir: Path, days: int) -> dict:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    run_dir = nemo_run_dir.resolve()
    require(run_dir.is_dir(), f"missing NEMO run directory {run_dir}")
    input_hashes = validate_run_input_hashes(run_dir)
    echo = parse_namelist_groups(run_dir / "output.namelist.dyn")
    reference = parse_namelist_groups(run_dir / "namelist_ref")
    config = parse_namelist_groups(run_dir / "namelist_cfg")
    driver = _load_driver()
    namelist = driver.resolved_daily_namelist(days)
    card = build_swg_card(namelist)
    comparison = compare_resolved(echo, reference, config, namelist)
    crosschecks = crosscheck_fallback_groups(
        run_dir / "ocean.output", comparison)
    return {
        "schema": "swg-dg-cross-model-deck-guard-v2",
        "checkout_import": CHECKOUT_IMPORT,
        "nemo_run_dir": str(run_dir),
        "nemo_sources": {
            "echo": str((run_dir / "output.namelist.dyn").resolve()),
            "reference": str((run_dir / "namelist_ref").resolve()),
            "configuration": str((run_dir / "namelist_cfg").resolve()),
            "precedence": "namelist_ref then namelist_cfg override",
            "manifest_validated_input_sha256": input_hashes,
            "ocean_output_cross_checks": crosschecks,
        },
        "legoesm_days": days,
        "legoesm_card_name": card.case,
        "legoesm_namelist": namelist._asdict(),
        **comparison,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-run-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.json.exists(), f"refusing to overwrite {args.json}")
        payload = run(args.nemo_run_dir, args.days)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - fail-closed command boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        "SWG_DG_DECK_GUARD PASS "
        f"compiled_reads={payload['compiled_inventory_fields_resolved']}/131 "
        f"comparison_rows={payload['comparison_rows']} "
        f"allowed_differences={len(payload['differences'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
