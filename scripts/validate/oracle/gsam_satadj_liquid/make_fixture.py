"""Generate the gSAM liquid saturation-adjustment oracle fixture by RUNNING
the oracle.

Oracle: gSAM 1.8.8 ``satadj_liquid`` (SRC/MICRO_M2005/microphysics.f90
:1638-1720), body verbatim, stock tolerance (thresh_mult = 1, maxiter = 20),
legoESM saturation curve fed in; built by
``scripts/validate/oracle/sb2001_warm_rain/build.sh`` (same gSAM driver).  No
git tag/SHA exists; the pin is the tarball
http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz (sha256
0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c).
The driver maps (T, q_v, q_c) to micro_proc's inputs q_t = q_v + q_c and
T_l = T - (L_v/c_pd) q_c, as micro_proc does.

Points: the 2026-10-19 tier-2 audit grid (T 230-310 K, p 40-1000 hPa,
supersaturated cloud-free points S = 1.0005-1.5 and sub/near-saturated cloudy
points), including p = 40 hPa where gSAM does not adjust.

Usage (CPU node; needs gfortran and the gSAM source)::

    python scripts/validate/oracle/gsam_satadj_liquid/make_fixture.py \\
        --gsam <dir containing SRC/MICRO_M2005> --build <dir> \\
        --out tests/atmosphere/microphysics/oracle_fixtures/gsam_satadj_liquid.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD_SH = HERE.parent / "sb2001_warm_rain" / "build.sh"
TEMPS = (230.0, 255.0, 273.0, 285.0, 300.0, 310.0)
PRESS = (4000.0, 20000.0, 50000.0, 85000.0, 100000.0)
SUPERSAT = (1.0005, 1.01, 1.05, 1.2, 1.5)
CLOUDY = ((0.8, 1e-5), (0.8, 1e-3), (0.95, 1e-4), (0.99, 1e-3), (0.999, 1e-5),
          (1.0, 1e-4))


def build_points():
    """Rows (T, q_v, q_c, p)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.thermo import saturation_specific_humidity
    rows = []
    for T in TEMPS:
        for p in PRESS:
            qs = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
            for S in SUPERSAT:
                rows.append([T, S * qs, 0.0, p])
            for S, qc in CLOUDY:
                rows.append([T, S * qs, qc, p])
    return np.array(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsam", required=True, help="dir containing SRC/MICRO_M2005")
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bld = Path(a.build)
    subprocess.run(["bash", str(BUILD_SH), a.gsam, str(bld)], check=True)
    pts = build_points()
    fin, fout = bld / "satadj_in.txt", bld / "satadj_out.txt"
    with open(fin, "w") as f:
        f.write(f"{len(pts)}\n")
        for r in pts:
            f.write(" ".join(f"{v:.17e}" for v in r) + "\n")
    log = subprocess.run([str(bld / "gsam_driver"), "satadj", "lego", "1", "20",
                          str(fin), str(fout)], check=True, capture_output=True,
                         text=True).stdout
    assert "SATADJ_OK" in log, log
    o = np.loadtxt(fout)
    assert o.shape == (len(pts), 4), o.shape
    gf = subprocess.run(["gfortran", "--version"], capture_output=True,
                        text=True).stdout.splitlines()[0]
    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    meta = dict(
        oracle="gSAM 1.8.8 SRC/MICRO_M2005/microphysics.f90 satadj_liquid",
        tarball="http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz",
        sha256="0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c",
        settings="stock tolerance thresh_mult=1, maxiter=20; legoESM q_sat curve",
        gfortran=gf, legoesm_sha=sha, driver_log=log.strip(),
        generator="python scripts/validate/oracle/gsam_satadj_liquid/make_fixture.py "
                  "--gsam <gSAM1.8.8> --build <dir> --out gsam_satadj_liquid.npz")
    np.savez_compressed(
        a.out, meta=json.dumps(meta), T=pts[:, 0], q_v=pts[:, 1], q_c=pts[:, 2],
        p=pts[:, 3], ora_T=o[:, 0], ora_q_v=o[:, 1], ora_q_c=o[:, 2],
        ora_maxiter_hit=o[:, 3])
    print("FIXTURE_OK", a.out, len(pts))


if __name__ == "__main__":
    main()
