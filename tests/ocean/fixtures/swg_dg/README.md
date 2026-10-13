# SWG_DG portable fixtures

These text fixtures replace tests that previously depended on one developer's
NEMO checkout. `namelist_cfg.shipped` and `usrdef_zgr.F90.shipped` are from
NEMO 5.0.2 revision `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`, relative
paths `tests/SWG/EXPREF/namelist_cfg` and
`tests/SWG/MY_SRC/usrdef_zgr.F90`. `output.namelist.dyn` is NEMO's resolved
echo from the pinned three-year Maddison run. They are test inputs only, not
model outputs cited as fresh Round-C acquisition evidence.

SHA-256:

- `namelist_cfg.shipped`: `6cc239d380f81be628348f655f5d0dca852c6d458f1459b05444293095a3ff7a`
- `usrdef_zgr.F90.shipped`: `14f6640b6720d5bda4cd7542dde65dd0add5aebd1d262000ed3b492172b5e06b`
- `output.namelist.dyn`: `a8f53c412dddb65128899c5f47ab7e82a4a5266c3a0d2929ce91a1d60f1970d3`
