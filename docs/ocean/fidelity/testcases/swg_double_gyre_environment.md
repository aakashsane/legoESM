# SWG double-gyre reference environment

Bit identity was measured with the complete composition below. The
machine-readable inventory is `swg_double_gyre_environment.json`; measured
runtime evidence is in `swg_double_gyre_reference_costs.json`.

## Reconstruct NEMO

The source is the public NEMO repository, tag `5.0.2`, at full revision
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`:

```bash
git clone --branch 5.0.2 --single-branch \
  https://forge.nemo-ocean.eu/nemo/nemo.git "$NEMO_ROOT"
git -C "$NEMO_ROOT" checkout --detach \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
test "$(git -C "$NEMO_ROOT" rev-parse HEAD)" = \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
```

For Linux x86-64, reproduce the exact conda-forge package builds used here:

```bash
conda create --prefix "$NEMO_PREFIX" --file \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/nemo-build-linux-64-explicit.txt"
```

That explicit file is normative for bit-identity attempts. The companion
`nemo-build-environment.yml` omits build strings and is a readable, more
portable fallback when exact linux-64 artifacts are unavailable:

```bash
conda env create --prefix "$NEMO_PREFIX" --file \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/nemo-build-environment.yml"
```

The reference environment has GNU Fortran 15.2.0 build 19, Open MPI 5.0.10
`h67ed482_1`, HDF5 2.1.0 `mpi_openmpi_hb23c26a_9`, libnetcdf 4.10.0
`mpi_openmpi_h75687c8_4`, and netcdf-fortran 4.6.3
`mpi_openmpi_h8f6c4f5_0`. XIOS is not used. The committed architecture uses
scalar math and these material flags:

```text
FCFLAGS -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer
        -ffree-line-length-none -fallow-argument-mismatch
        -fno-tree-vectorize
CPP     cpp -Dkey_nosignedzero
CFLAGS  -O0
```

`build_swg_dg.sh` renders the conda prefix into the architecture and records
the resulting architecture, binary, case sources, compiler versions, flags,
and hashes in its self-contained build manifest.

## Reconstruct Python/JAX

Create a Python 3.13.0 virtual environment and install the restricted,
fully-pinned dependency closure used by the driver, comparators, climate
instrument, and movie:

```bash
python3.13 -c 'import sys; assert sys.version_info[:3] == (3, 13, 0)'
python3.13 -m venv "$PYTHON_ENV"
PY="$PYTHON_ENV/bin/python"
"$PY" -m pip install -r \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/python-gpu-requirements.txt"
```

The pin includes JAX, jaxlib, `jax-cuda12-plugin`, and
`jax-cuda12-pjrt` 0.10.0; NumPy 2.4.4; Equinox 0.13.7; every CUDA wheel build;
NetCDF4; SciPy; Matplotlib; and imageio-ffmpeg. The package tools add their own
checkout's package roots before imports and print `CHECKOUT_IMPORT_OK` with
the selected Python, NumPy, and `legoesm.ocean` paths. They refuse an import
from another checkout. No editable legoESM install is required.

## Reference machine and capacity

| Component | Measured reference |
|---|---|
| GPU | Tesla V100S-PCIE-32GB; driver 580.126.20 |
| CPU | Intel Xeon Gold 6242R |
| OS | Linux 5.14.0-611.36.1.el9_7.x86_64 |
| NEMO three-year storage | 56,985,134,712 bytes |
| legoESM daily-mean storage | 21,803,971,913 bytes |

Reserve at least 100 GiB of free output space. A 32 GiB GPU is the minimum
tested capacity, not a measured lower bound: smaller GPUs may work, but have
not been admitted. The Round-C driver records both `wall_seconds` and JAX's
`peak_bytes_in_use` allocator statistic in `manifest.json` and refuses a GPU
backend that cannot report that high-water value.

Exact floating-point identity is conditional on compiler versions and flags,
the scalar-math architecture, `key_nosignedzero`, JAX/XLA versions, and the
selected CPU/GPU program composition. Always retain build/run manifests with
comparison reports.
