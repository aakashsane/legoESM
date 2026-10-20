"""Write the legoESM-fed scratch copy of gSAM 1.8.8 MICRO_P3/module_mp_p3.f90.

Usage: python3 patch_p3.py <dir with module_mp_p3.f90 + micro_params.f90> <out_dir>

Same patch as the tier-2 p3_microphysics audit (oracle_audit/p3_microphysics/py/
patch_p3.py, "lego" variant).  Unified diffs are written alongside.

module_mp_p3_lego.f90:
  * qv_sat -> legoESM saturation SPECIFIC humidity (lego_sat stub in
    stubs_common.inc; liquid AND ice; the stub reproduces legoesm.thermo to
    4e-16 / 6e-15, checked by make_fixture.py).  polysvp1 (used once, for
    e0 = polysvp1 at the freezing point = 100*esatw, SAM sat.f90) is NOT patched.
  * process-rate dumps into diag_3d (instrumentation only, no arithmetic change).
  * construct label rename (gfortran 8 compile fix, no arithmetic change).
micro_params_lego.f90: typed CHARACTER array constructors (gfortran 8 compile
  fix; diagnostic labels only).

diag_3d slots (rates AFTER the conservation limiters, i.e. the values that update
the state; [kg/kg/s] or [#/kg/s]):
  1 qccon 2 qcevp 3 qrcon 4 qrevp 5 qcaut 6 qcacc 7 ncautr 8 nrslf 9 qccol
  10 qrcol 11 nrcol 12 qidep 13 qisub 14 qinuc 15 ninuc 16 nislf 17 qimlt
  18 nimlt 19 qcheti 20 qrheti 21 qcshd 22 qwgrth 23 rhorime_c 24 nccol
  25 nrevp 26 qcnuc 27 ncheti 28 qchomi(rate) 29 qrhomi(rate) 30 nrheti
  31 ncautc 32 ncacc 33 nisub 34 epsi_tot(stale on levels without hydrometeors)
  35 qinuc2+qinuc3 36 ncshdc
"""
import difflib
import os
import sys

src, out = sys.argv[1], sys.argv[2]
orig = open(os.path.join(src, "module_mp_p3.f90")).read()

ANCHOR_RATES = "       iice_loop2: do iice = 1,nCat\n"
DUMP = """!ORACLE-FIXTURE begin: process-rate dumps (instrumentation only)
       diag_3d(i,k,1) = qccon;  diag_3d(i,k,2) = qcevp;  diag_3d(i,k,3) = qrcon
       diag_3d(i,k,4) = qrevp;  diag_3d(i,k,5) = qcaut;  diag_3d(i,k,6) = qcacc
       diag_3d(i,k,7) = ncautr; diag_3d(i,k,8) = nrslf;  diag_3d(i,k,9) = sum(qccol)
       diag_3d(i,k,10) = sum(qrcol); diag_3d(i,k,11) = sum(nrcol); diag_3d(i,k,12) = sum(qidep)
       diag_3d(i,k,13) = sum(qisub); diag_3d(i,k,14) = sum(qinuc); diag_3d(i,k,15) = sum(ninuc)
       diag_3d(i,k,16) = sum(nislf); diag_3d(i,k,17) = sum(qimlt); diag_3d(i,k,18) = sum(nimlt)
       diag_3d(i,k,19) = sum(qcheti); diag_3d(i,k,20) = sum(qrheti); diag_3d(i,k,21) = sum(qcshd)
       diag_3d(i,k,22) = sum(qwgrth); diag_3d(i,k,23) = rhorime_c(1); diag_3d(i,k,24) = sum(nccol)
       diag_3d(i,k,25) = nrevp; diag_3d(i,k,26) = qcnuc; diag_3d(i,k,27) = sum(ncheti)
       diag_3d(i,k,30) = sum(nrheti); diag_3d(i,k,31) = ncautc; diag_3d(i,k,32) = ncacc
       diag_3d(i,k,33) = sum(nisub); diag_3d(i,k,34) = epsi_tot
       diag_3d(i,k,35) = sum(qinuc2)+sum(qinuc3); diag_3d(i,k,36) = sum(ncshdc)
!ORACLE-FIXTURE end
"""
ANCHOR_HOM = "         qcrfrz_hom=SUM(qrhomi(:))+SUM(qchomi(:))\n"
DUMP_HOM = ANCHOR_HOM + """!ORACLE-FIXTURE: homogeneous-freezing dumps (qchomi already /dt; qrhomi is a mass)
         diag_3d(i,k,28) = SUM(qchomi(:)); diag_3d(i,k,29) = SUM(qrhomi(:))*odt
"""
ANCHOR_QSAT = " e_pres = polysvp1(t_atm,i_wrt)\n qv_sat = ep_2*e_pres/max(1.e-3,(p_atm-e_pres))\n"
LEGO_QSAT = """!ORACLE-FIXTURE: legoESM saturation SPECIFIC humidity (thermo.py) fed to the oracle
 if (i_wrt == 1) then
    qv_sat = qsi_lego(t_atm, p_atm)
 else
    qv_sat = qsl_lego(t_atm, p_atm)
 endif
"""
for anc in (ANCHOR_RATES, ANCHOR_HOM, ANCHOR_QSAT):
    assert orig.count(anc) == 1, anc

lego = orig.replace(ANCHOR_RATES, DUMP + ANCHOR_RATES).replace(ANCHOR_HOM, DUMP_HOM)
lego = lego.replace(ANCHOR_QSAT, LEGO_QSAT).replace(
    "use grid, only: masterproc, dompi,nstep,rank\n",
    "use grid, only: masterproc, dompi,nstep,rank\nuse lego_sat, only: qsl_lego, qsi_lego !ORACLE-FIXTURE\n")
assert lego.count("use lego_sat") == 1
# gfortran 8: construct label 'doSizeDependentIceFallFudge' collides with the
# micro_params logical of the same name -> rename the LABEL only.
LBL_A = "doSizeDependentIceFallFudge: if(doSizeDependentIceFallFudge) then"
LBL_B = "end if doSizeDependentIceFallFudge"
assert orig.count(LBL_A) == 1 and orig.count(LBL_B) == 1
lego = lego.replace(LBL_A, "sdiff_lbl: if(doSizeDependentIceFallFudge) then").replace(
    LBL_B, "end if sdiff_lbl")
with open(os.path.join(out, "module_mp_p3_lego.f90"), "w") as f:
    f.write(lego)
with open(os.path.join(out, "module_mp_p3_lego.f90.diff"), "w") as f:
    f.writelines(difflib.unified_diff(orig.splitlines(True), lego.splitlines(True),
                                      "module_mp_p3.f90", "module_mp_p3_lego.f90"))

mp = open(os.path.join(src, "micro_params.f90")).read()
a1 = "        micro_process_rate_names = (/ &\n"
a2 = "        micro_process_rate_longnames = (/ &\n"
assert mp.count(a1) == 1 and mp.count(a2) == 1
mp2 = mp.replace(a1, "        micro_process_rate_names = (/ character(len=8) :: &\n").replace(
    a2, "        micro_process_rate_longnames = (/ character(len=80) :: &\n")
with open(os.path.join(out, "micro_params_lego.f90"), "w") as f:
    f.write(mp2)
with open(os.path.join(out, "micro_params_lego.f90.diff"), "w") as f:
    f.writelines(difflib.unified_diff(mp.splitlines(True), mp2.splitlines(True),
                                      "micro_params.f90", "micro_params_lego.f90"))
print("PATCH_OK")
