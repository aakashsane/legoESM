"""Bind a standalone NEMO-fidelity entry point to its owning checkout."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PACKAGE_ROOTS = (
    "packages/core",
    "packages/ocean",
    "packages/atmosphere",
    "packages/coupler",
    "packages/ice",
    "packages/land",
    "packages/ml",
    "packages/tools",
    "src",
)


def bootstrap_checkout(script: str | Path) -> dict[str, str]:
    """Put the script's Git checkout first and verify the imported package."""
    path = Path(script).resolve()
    try:
        root = Path(subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot locate the Git checkout owning {path}: {error}"
        ) from error
    roots = [str(root / relative) for relative in _PACKAGE_ROOTS]
    sys.path[:0] = roots
    from legoesm.ocean.fidelity.provenance import require_checkout_import

    return require_checkout_import(root)
