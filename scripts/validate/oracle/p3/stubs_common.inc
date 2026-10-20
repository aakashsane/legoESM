module grid
  implicit none
  logical :: masterproc = .true., dompi = .false.
  integer :: nstep = 2, rank = 0
end module grid

module lego_sat
  implicit none
  real, parameter, private :: rd = 287.05, rv_ = 461.51, tf = 273.15
  real, parameter, private :: ls = 2.501e6 + 3.337e5, eps = rd/rv_
  interface
     pure function c_log1p(x) bind(C, name='log1p')
       use iso_c_binding, only: c_double
       real(c_double), value :: x
       real(c_double) :: c_log1p
     end function c_log1p
  end interface
contains
  elemental real function softplus(x)
    real, intent(in) :: x
    softplus = max(x, 0.) + c_log1p(exp(-abs(x)))
  end function softplus
  elemental real function qs_from_e(e, p)
    real, intent(in) :: e, p
    real :: w
    w = eps*e/(softplus(p - e - 1.) + 1.)
    w = 1. - softplus(20.*(1. - w))/20.
    qs_from_e = w/(1. + w)
  end function qs_from_e
  elemental real function qsl_lego(t, p)
    real, intent(in) :: t, p
    real :: tc
    tc = max(t, 150.) - tf
    qsl_lego = qs_from_e(611.2*exp(17.67*tc/(tc + 243.5)), p)
  end function qsl_lego
  elemental real function qsi_lego(t, p)
    real, intent(in) :: t, p
    qsi_lego = qs_from_e(611.2*exp(ls/rv_*(1./tf - 1./t)), p)
  end function qsi_lego
end module lego_sat

subroutine task_abort()
  stop 9
end subroutine task_abort
subroutine task_abort_msg(msg)
  character(*) :: msg
  print*, msg
  stop 9
end subroutine task_abort_msg
subroutine task_bcast_integer(a,b,c)
  integer a, b(*), c
  stop 8
end subroutine task_bcast_integer
subroutine task_bcast_fourdim_array_real(a,b,c,d,e,f)
  integer a,c,d,e,f
  real b(*)
  stop 8
end subroutine task_bcast_fourdim_array_real
subroutine task_bcast_fivedim_array_real8(a,b,c,d,e,f,g)
  integer a,c,d,e,f,g
  real(8) b(*)
  stop 8
end subroutine task_bcast_fivedim_array_real8
subroutine task_bcast_sixdim_array_real8(a,b,c,d,e,f,g,h)
  integer a,c,d,e,f,g,h
  real(8) b(*)
  stop 8
end subroutine task_bcast_sixdim_array_real8
