"""ExperimentConfig.surface_bulk_scheme="none": no surface bulk exchange.

Validation, the YAML/CLI mapping, and the single-device lat-lon pipeline
honouring it (it used to apply constant-coefficient fluxes even with every
physics scheme off, while the multi-device lat-band lane applied none).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig, VALID_SURFACE_BULK)

_REPO = Path(__file__).resolve().parents[2]
_DT = 100.0
_DRY = dict(turbulence="none", radiation="none", convection="none",
            microphysics="none", gravity_wave_drag="none", cloud_scheme="none")


def test_none_is_a_valid_scheme_on_a_dry_config():
    assert "none" in VALID_SURFACE_BULK
    ExperimentConfig(surface_bulk_scheme="none", **_DRY).validate_strict()


@pytest.mark.parametrize("over", [
    {"turbulence": "louis"},
    {"physics_parameterization": "ml"},
    {"slab_land_active": True},
    {"use_multilayer_land": True},
    {"surface_tiled": True, "turbulence": "louis"},
])
def test_none_rejected_next_to_a_surface_flux_owner(over):
    cfg = ExperimentConfig(surface_bulk_scheme="none", **{**_DRY, **over})
    with pytest.raises(ValueError, match="surface_bulk_scheme='none'"):
        cfg.validate_strict()


def test_held_suarez_template_selects_none():
    from legoesm.config import Config
    cfg = Config.from_yaml(
        _REPO / "config/templates/3d_idealized/held_suarez.yaml"
    ).to_experiment_config()
    assert cfg.surface_bulk_scheme == "none"


def test_run_amip_cli_round_trips_none():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args)
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--surface-bulk-scheme", "none"]), parser)
    assert build_config_from_args(args).surface_bulk_scheme == "none"


def _one_step_T(tmp_path, sbs):
    """One Held-Suarez step on the single-device lat-lon driver, and the same
    step taken by the bare dycore with only the Held-Suarez forcing."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_forcing_latlon)
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT, fix_mass=True),
        output=OutputConfig(output_dir=str(tmp_path / sbs), diag_days=0,
                            checkpoint_days=0),
        days=_DT / 86400.0, dataset="analytical", held_suarez_forcing=True,
        fix_moisture=True, enable_latlon_spmd=False, distributed=False,
        n_devices=1, surface_bulk_scheme=sbs, **_DRY)
    d = ModelDriver(cfg, output_dir=cfg.output.output_dir)
    d.setup()
    bare = d.model.step(d.state, _DT, physics_fn=held_suarez_forcing_latlon)
    assert d.run(compiled=True) == "COMPLETED"
    return (np.asarray(d.state.T.data, np.float64),
            np.asarray(bare.T.data, np.float64))


def test_single_device_driver_applies_no_surface_flux_under_none(tmp_path):
    drv, bare = _one_step_T(tmp_path, "none")
    # One fp32 rounding step at ~300 K is 3.05e-5 K; the constant-coefficient
    # bulk kick this replaces is ~9e-3 K at the bottom level.
    assert np.abs(drv - bare).max() < 1e-4
    drv_c, bare_c = _one_step_T(tmp_path, "constant")
    assert np.abs(drv_c - bare_c).max() > 1e-3
