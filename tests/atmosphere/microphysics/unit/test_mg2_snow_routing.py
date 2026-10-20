"""CAM6 MG2 mode decisions (user 2026-10-20): Bigg-frozen rain becomes snow,
or cloud ice where the mean drop is smaller than DCS, never graupel
(micro_mg2_0.F90:1648-1656, :1882-1895); DCS is 250 um for the SAM flavour
and 500 um for the MG flavour; lanes that cannot carry the snow number refuse
two-moment snow."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mg2_warm_rain_oracle import _production_morrison  # noqa: E402

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import (  # noqa: E402
    morrison_microphysics, resolve_morrison_flavor)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402


def test_dcs_per_flavour():
    sam = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="sam"))
    mg = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="mg"))
    assert sam.ice_snow_d_auto == 250.0e-6
    assert mg.ice_snow_d_auto == 500.0e-6
    kept = resolve_morrison_flavor(
        MorrisonConfig(morrison_flavor="mg", ice_snow_d_auto=300.0e-6))
    assert kept.ice_snow_d_auto == 300.0e-6


def _freeze_rain(cfg, N_r_per_m3):
    """One cold, rainy, cloud-free, subsaturated level: only Bigg freezing
    (and sublimation/evaporation) act on the rain."""
    T, p, rho = 255.0, 5.0e4, 0.68
    one = jnp.ones((1, 1))
    z = jnp.zeros((1, 1))
    hyd = HydrometeorState(q_c=z, q_r=1.0e-3 * one, q_i=z, q_s=z, q_g=z,
                           N_c=z, N_r=N_r_per_m3 * one, N_i=z, N_s=z)
    p_half = jnp.array([[p - 1.0e3, p + 1.0e3]])
    dz = (p_half[:, 1:] - p_half[:, :-1]) / (C.g * rho)
    return morrison_microphysics(
        T * one, 1.0e-4 * one, hyd, p * one, p_half, rho * one, dz, 60.0,
        cfg, cloud_fraction=z)


def test_mg2_frozen_rain_never_goes_to_graupel():
    cfg = _production_morrison()
    assert cfg.do_graupel and cfg.warm_rain_incloud
    out = _freeze_rain(cfg, 1.0e6)            # small drops: 1/lamr < DCS
    assert float(out.dq_r_dt[0, 0]) < 0.0
    assert float(out.dq_g_dt[0, 0]) == 0.0
    assert float(out.dq_i_dt[0, 0]) > 0.0     # MG2 mnuccri: frozen small drops -> ice
    assert float(out.dN_i_dt[0, 0]) > 0.0


def test_mg2_frozen_large_drops_go_to_snow():
    # explicit DCS below the mean drop size: the frozen rain becomes snow
    cfg = _production_morrison()._replace(ice_snow_d_auto=20.0e-6)
    out = _freeze_rain(cfg, 1.0e6)
    assert float(out.dq_g_dt[0, 0]) == 0.0
    assert float(out.dq_s_dt[0, 0]) > 0.0
    assert float(out.dN_s_dt[0, 0]) > 0.0
    assert float(out.dq_i_dt[0, 0]) <= 0.0


@pytest.mark.parametrize("grid,disc", [("latlon", "cdgrid"),
                                       ("cubed_sphere", "fv3"),
                                       ("mpas", "fv3_duo")])
def test_lanes_without_snow_number_refuse_mg2_snow(grid, disc):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type=grid), dycore=DycoreConfig(discretization=disc),
        microphysics="morrison", morrison_warm_rain_incloud=True,
        turbulence="clubb", cloud_scheme="cam6_clubb",
        use_clubb_cloud_fraction=True, cld_macmic_num_steps=2,
        subgrid_autoconversion=False)
    with pytest.raises(ValueError, match="no other lane carries the snow number"):
        cfg.validate_strict()


def test_mpas_lane_accepts_mg2_snow():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas"), dycore=DycoreConfig(discretization="mpas"),
        microphysics="morrison", morrison_warm_rain_incloud=True,
        turbulence="clubb", cloud_scheme="cam6_clubb",
        use_clubb_cloud_fraction=True, cld_macmic_num_steps=2,
        subgrid_autoconversion=False)
    try:
        cfg.validate_strict()
    except ValueError as e:
        assert "no other lane carries the snow number" not in str(e)
