"""Generate tests/atmosphere/oracle_fixtures/ysu.npz by RUNNING the WRF YSU
oracle (WRF v4.8.0 phys/physics_mmm/bl_ysu.F90::bl_ysu_run, clear-sky path:
topdown off, qc=qi=0, no BEP, ctopo absent).

usage:
  bash build.sh <WRF> <build>
  python make_fixture.py --inputs <columns.npz> --driver <build>/ysu_driver \
      --out tests/atmosphere/oracle_fixtures/ysu.npz

``--inputs`` is the tier-2 audit column set (oracle_audit/ysu_pbl/inputs.npz:
20 L32 columns, real MPAS checkpoint columns + constructed soundings, with
the audit's constructed surface-layer state; sha256 recorded in ``meta``).
Two more columns are appended: copies of the stable land columns
``polar_dry`` and ``nocturnal_elevated`` with ``xland = 2`` (water), so the
fixture reaches the oracle's Rossby-number critical-Ri branch over water.
Constants handed to the oracle are legoESM's (``legoesm.constants``), as
``bl_ysu_run`` (the JAX port) uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants as C

ORACLE = dict(repo="wrf-model/WRF", tag="v4.8.0",
              sha="06d4240ae989cc3e50af412bb472df3d9048783c",
              physics_mmm_sha="550b5b494758ad2756394e93cb08b5709b8feb83",
              files=["phys/physics_mmm/bl_ysu.F90", "phys/ccpp_kind_types.F"])
LEV = ("u", "v", "T", "q", "p_full", "exner", "dz")
SFC = ("ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "znt", "xland", "u10", "v10")
WATER_COPIES = ("polar_dry", "nocturnal_elevated")


def constants_vector(dt):
    # cp g rovcp rd rovg ep1 ep2 karman xlv rv dt (ysu_driver.F90 order)
    return [C.c_pd, C.g, C.R_d / C.c_pd, C.R_d, C.R_d / C.g, C.R_v / C.R_d - 1.0,
            C.R_d / C.R_v, C.kappa_vk, C.L_v, C.R_v, dt]


def columns(src):
    names = [str(n) for n in src["names"]]
    d = dict(u=src["u"], v=src["v"], T=src["T"], q=src["q"], p_full=src["p_full"],
             p_half=src["p_half"], zi=src["zi"], ust=src["ust"], hfx=src["shflx"],
             qfx=src["qflx"], wspd=src["wspd"], br=src["br"],
             psim=src["psim"], psih=src["psih"],     # audit stores fm, fh under these names
             znt=src["z0"], xland=src["xland"], u10=src["u10"], v10=src["v10"],
             taux=src["taux"], tauy=src["tauy"], lhflx=src["lhflx"])
    d = {k: np.asarray(v, float) for k, v in d.items()}
    idx = [names.index(n) for n in WATER_COPIES]
    d = {k: np.concatenate([v, v[idx]], axis=0) for k, v in d.items()}
    d["xland"][-len(idx):] = 2.0
    names = names + [f"{n}_water" for n in WATER_COPIES]
    d["exner"] = (d["p_full"] / C.p_ref) ** C.kappa
    d["dz"] = d["zi"][:, :-1] - d["zi"][:, 1:]
    return names, d


def write_inputs(path, d, dt):
    def wr(f, a):
        f.write(" ".join(repr(float(x)) for x in np.ravel(a)) + "\n")
    ncol, nlev = d["T"].shape
    with open(path, "w") as f:
        f.write(f"{ncol} {nlev}\n")
        wr(f, constants_vector(dt))
        for c in range(ncol):
            for k in LEV:
                wr(f, d[k][c][::-1])
            wr(f, d["p_half"][c][::-1])
            wr(f, [d["p_half"][c, -1], d["znt"][c], d["ust"][c], d["hfx"][c], d["qfx"][c],
                   d["wspd"][c], d["br"][c], d["psim"][c], d["psih"][c], d["xland"][c],
                   d["u10"][c], d["v10"][c]])


def parse_outputs(path, ncol):
    lines = open(path).read().split("\n")
    assert lines[0].startswith("YSU") and "bl_ysu_run OK" in lines[0], lines[0]
    o = {k: [] for k in ("kpbl", "hpbl", "wstar", "delta", "du", "dv", "dT", "dq",
                         "exch_h", "exch_m")}
    i = 1
    for c in range(ncol):
        head = lines[i].split()
        assert head[:2] == ["COL", str(c + 1)], lines[i]
        o["kpbl"].append(int(head[2])); i += 1
        s = np.array(lines[i].split(), float); i += 1
        o["hpbl"].append(s[0]); o["wstar"].append(s[1]); o["delta"].append(s[2])
        for k in ("du", "dv", "dT", "dq", "exch_h", "exch_m"):
            o[k].append(np.array(lines[i].split(), float)[::-1]); i += 1   # -> top-first
    return {f"ora_{k}": np.array(v) for k, v in o.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = np.load(a.inputs)
    dt = float(src["dt"])
    names, d = columns(src)
    ncol = d["T"].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        write_inputs(fin, d, dt)
        r = subprocess.run([a.driver, str(fin), str(fout), "0"], capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0, r
        ora = parse_outputs(fout, ncol)
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran + " -O2 -DDOUBLE_PRECISION -fdefault-real-8",
                constants=dict(zip(("cp", "g", "rovcp", "rd", "rovg", "ep1", "ep2", "karman",
                                    "xlv", "rv", "dt"), constants_vector(dt))),
                inputs_sha256=hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
                names=names, water_copies=list(WATER_COPIES),
                setup="ysu_topdown_pblmix=F, f_qc=f_qi=F, nmix=0, no BEP, ctopo/uox absent",
                layout="arrays top-first (index 0 = model top); exch_* entry k = K of the "
                       "interface below level k (oracle exch(k+1)=xkz(k)); kpbl 1-based "
                       "from the surface")
    np.savez_compressed(a.out, meta=json.dumps(meta), dt=dt,
                        **{f"in_{k}": v for k, v in d.items()}, **ora)
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
