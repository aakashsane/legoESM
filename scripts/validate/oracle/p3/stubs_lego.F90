! STUBS for compiling gSAM 1.8.8 SRC/MICRO_P3/module_mp_p3.f90 standalone (audit p3_microphysics).
! params: legoESM constants (packages/core/legoesm/constants.py @ wt_oracle_main 37915678bb):
!   cp = c_pd = 1004.64, lcond = L_v = 2.501e6, lfus = L_f = 3.337e5 (=> P3 xxls = L_v+L_f = L_s),
!   rgas = R_d = 287.05, rv = R_v = 461.51, ggr = g = 9.80616.
!   gSAM consts.f90 values (stubs_gsam.F90): cp 1004.64, lcond 2.501e6, lfus 0.337e6, rgas 287.04,
!   rv 461.5, ggr 9.79764.
! grid: masterproc/dompi/nstep/rank only (dompi=.false. => the task_bcast_* calls never execute).
! lego_sat: legoESM saturation SPECIFIC humidity (thermo.saturation_specific_humidity[_ice]); same
!   code as the tier-2 warm_rain_satadj audit stub; re-verified against legoesm.thermo by harness.py.
module params
  implicit none
  real(8), parameter :: cp = 1004.64d0, lcond = 2.501d6, lfus = 3.337d5
  real(8), parameter :: rgas = 287.05d0, rv = 461.51d0, ggr = 9.80616d0
end module params
#include "stubs_common.inc"
