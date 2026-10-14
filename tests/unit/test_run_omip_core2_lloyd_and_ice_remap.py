"""Two OMIP driver plumbing defects (review 2026-10-14, Round C).

1. ``--mpas-lloyd`` was inert: ``build_mpas_ocean`` took ``lloyd_iterations``
   but never handed it to ``run_omip._create_setup``, so every MPAS run built
   the setup default of 50 Lloyd iterations while the flag defaulted to 20.
   User decision 2026-10-12: cards state 50.  The flag now reaches the mesh
   builder and defaults to 50, so the default mesh is the one runs already used.
2. F19 residual: the prognostic sea-ice forcing sample took no
   ``forcing_remap``, so under ``--forcing-remap nemo_scrip`` the ice saw
   bilinear fields while the ocean saw NEMO's SCRIP remap.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest


class _Stop(Exception):
    pass


def _capture_create_setup(monkeypatch):
    from scripts.run import run_omip
    captured = {}

    def _fake(*a, **k):
        captured.update(k)
        raise _Stop

    monkeypatch.setattr(run_omip, "_create_setup", _fake)
    return captured


def test_mpas_lloyd_flag_defaults_to_the_mesh_runs_already_built():
    import scripts.run.run_omip_core2 as core2
    p = core2._build_arg_parser()
    assert p.parse_args(["--grid", "mpas"]).mpas_lloyd == 50
    assert p.parse_args(["--grid", "mpas", "--mpas-lloyd", "7"]).mpas_lloyd == 7


@pytest.mark.parametrize("lloyd", [0, 7, 50])
def test_build_mpas_ocean_forwards_lloyd_to_the_mesh_builder(monkeypatch, lloyd):
    import scripts.run.run_omip_core2 as core2
    captured = _capture_create_setup(monkeypatch)
    with pytest.raises(_Stop):
        core2.build_mpas_ocean(5, 5500.0, "dummy", level=3,
                               lloyd_iterations=lloyd)
    assert captured.get("mpas_lloyd") == lloyd


def test_build_mpas_ocean_default_lloyd_is_50(monkeypatch):
    import scripts.run.run_omip_core2 as core2
    captured = _capture_create_setup(monkeypatch)
    with pytest.raises(_Stop):
        core2.build_mpas_ocean(5, 5500.0, "dummy", level=3)
    assert captured.get("mpas_lloyd") == 50


def test_public_sampler_forwards_forcing_remap(monkeypatch):
    from legoesm.ocean.coupler import omip2_applicator as m
    seen = {}

    def _fake(forcing, idx_t, grid, grid_type, forcing_remap="bilinear"):
        seen["remap"] = forcing_remap
        return {}

    monkeypatch.setattr(m, "_sample_omip2_forcing", _fake)
    m.sample_omip2_forcing(None, 0, None, "tripole", forcing_remap="nemo_scrip")
    assert seen["remap"] == "nemo_scrip"
    m.sample_omip2_forcing(None, 0, None, "tripole")
    assert seen["remap"] == "bilinear"


def test_driver_ice_sample_uses_the_ocean_forcing_remap():
    """``main`` is the function that runs the prognostic-ice loop; every
    ``sample_omip2_forcing`` call in it must pass the SAME
    ``forcing_remap=args.forcing_remap`` the ocean forcing gets."""
    import scripts.run.run_omip_core2 as core2
    tree = ast.parse(textwrap.dedent(inspect.getsource(core2.main)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None))
             == "sample_omip2_forcing"]
    assert calls, "main no longer samples ice forcing; update this test"
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("forcing_remap") == "args.forcing_remap", ast.unparse(c)
