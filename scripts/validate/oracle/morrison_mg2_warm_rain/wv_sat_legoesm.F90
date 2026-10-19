! STUB of CAM wv_sat_methods (module name kept so micro_mg2_0.F90 compiles
! unmodified).  Reproduces legoESM's microphysics saturation curve EXACTLY
! (packages/core/legoesm/thermo.py, legoESM 37915678bb):
!   e_l  = 611.2*exp(17.67*Tc/(Tc+243.5)), Tc = max(T,150)-T_freeze   (Tetens/Bolton)
!   e_i  = 611.2*exp(L_s/R_v*(1/T_freeze - 1/T))
!   w    = 1 - softplus(20*(1 - eps*e/(softplus(p-e-1)+1)))/20      (smooth caps)
!   qs   = w/(1+w)                                       (specific humidity)
! with legoESM constants (R_d=287.05, R_v=461.51, L_v=2.501e6, L_f=3.337e5,
! T_freeze=273.15).  CAM default is Goff-Gratch: compile the REAL
! wv_sat_methods.F90 instead (-DREAL_WVSAT build) to measure that mismatch.
module wv_sat_methods
  implicit none
  private
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: rd = 287.05_r8, rv = 461.51_r8, tf = 273.15_r8
  real(r8), parameter :: ls = 2.501e6_r8 + 3.337e5_r8
  real(r8), parameter :: eps = rd/rv
  public :: wv_sat_qsat_water, wv_sat_qsat_ice
  interface
     pure function c_log1p(x) bind(C, name='log1p')
       use iso_c_binding, only: c_double
       real(c_double), value :: x
       real(c_double) :: c_log1p
     end function c_log1p
  end interface
contains
  elemental real(r8) function softplus(x)
    real(r8), intent(in) :: x
    softplus = max(x, 0._r8) + c_log1p(exp(-abs(x)))
  end function softplus
  elemental real(r8) function qs_from_e(e, p)
    real(r8), intent(in) :: e, p
    real(r8) :: w
    w = eps*e/(softplus(p - e - 1._r8) + 1._r8)
    w = 1._r8 - softplus(20._r8*(1._r8 - w))/20._r8
    qs_from_e = w/(1._r8 + w)
  end function qs_from_e
  elemental subroutine wv_sat_qsat_water(t, p, es, qs, idx)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    integer, intent(in), optional :: idx
    real(r8) :: tc
    tc = max(t, 150._r8) - tf
    es = 611.2_r8*exp(17.67_r8*tc/(tc + 243.5_r8))
    qs = qs_from_e(es, p)
    es = min(es, p)
  end subroutine wv_sat_qsat_water
  elemental subroutine wv_sat_qsat_ice(t, p, es, qs, idx)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    integer, intent(in), optional :: idx
    es = 611.2_r8*exp(ls/rv*(1._r8/tf - 1._r8/t))
    qs = qs_from_e(es, p)
    es = min(es, p)
  end subroutine wv_sat_qsat_ice
end module wv_sat_methods
