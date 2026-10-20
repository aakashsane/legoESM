! Standalone single-column driver for gSAM 1.8.8 MICRO_P3 p3_main (audit p3_microphysics).
! Mirrors the gSAM micro_proc call (SRC/MICRO_P3/microphysics.f90:869-895): model='WRF'
! (k=1 bottom), nCat=1, scpf_on=.false., clbfact_dep=clbfact_sub=1, debug_on=.false.,
! ssat=0, typeDiags_ON=.true.  Harness glue (NOT oracle physics): th_old=th and
! qv_old=qv (no dynamical forcing: the dT/dt and dqv/dt parts of P3's 'aaa' term are 0; its Bergeron (qvs-qvi)*epsi_tot part remains),
! uzpl=0, th = T*(1e5/p)**(rgas/cp) with the SAME rgas/cp P3 uses internally (gSAM's
! interface uses a hard 0.286; replaced so T round-trips).  Flags set at run time
! through the public micro_params variables BEFORE p3_init.
!
! input  (argv1): header 'ncol nlev dt it do_meyers iWarm predictNc Nc0 MaxTotalIceNumber iIceNuc'
!                 'lookup_dir'; then ncol*nlev lines bottom-up:
!                 p rho dz T qv qc nc qr nr qi qirim ni birim   (SI, numbers per kg)
! output (argv2): per column 'COL icol prt_liq prt_sol' then nlev lines:
!                 T qv qc nc qr nr qi qirim ni birim mtend(1:8) stend(1:8) diag(1:36)
program p3_driver
  use micro_params
  use params, only: cp, rgas
  use module_mp_p3, only: p3_init, p3_main
  implicit none
  integer, parameter :: ndiag = 36
  integer :: ncol, nlev, it, idm, ipnc, icol, k, stat, iwarm, inuc
  real :: dt, nc0_in, maxni
  character(len=1024) :: fin, fout, lookdir
  real, allocatable :: p(:,:), rh(:,:), dz(:,:), tt(:,:), qv(:,:), qc(:,:), nc(:,:), qr(:,:), &
       nr(:,:), qi(:,:,:), qir(:,:,:), ni(:,:,:), bir(:,:,:), th(:,:), th_old(:,:), qv_old(:,:), &
       ssat(:,:), uz(:,:), ze(:,:), effc(:,:), qcl(:,:), dcl(:,:), qsn(:,:), dsn(:,:), qgr(:,:), &
       dgr(:,:), effi(:,:,:), vmi(:,:,:), di(:,:,:), rhoi(:,:,:), d2(:,:), d3(:,:,:), scf(:,:), &
       qit(:,:,:), pg(:,:), lc(:,:), mt(:,:,:), st(:,:,:), mpr(:,:,:)
  real :: prl(1), prs(1), p1(1), p2(1), p3(1), p4(1), p5(1), p6(1), p7(1), p8(1)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(11, file=trim(fin), status='old')
  read(11,*) ncol, nlev, dt, it, idm, iwarm, ipnc, nc0_in, maxni, inuc
  read(11,'(a)') lookdir

  do_meyers = (idm == 1)
  iWarmRainScheme = iwarm
  log_predictNc = (ipnc == 1)
  Nc0 = nc0_in
  MaxTotalIceNumber = maxni
  iIceNucleationScheme = inuc
  nCat = 1
  nmicro_proc = nmicro_process_rates
  allocate(iqit(1), inci(1), iqir(1), iqib(1))
  iqv = 1; incl = 2; iqr = 3; inr = 4; iqit(1) = 5; inci(1) = 6; iqir(1) = 7; iqib(1) = 8; iqcl = 9

  call p3_init(trim(lookdir), 1, 'WRF', stat)
  if (stat /= 0) stop 'P3_INIT_FAILED'

  allocate(p(1,nlev), rh(1,nlev), dz(1,nlev), tt(1,nlev), qv(1,nlev), qc(1,nlev), nc(1,nlev), &
       qr(1,nlev), nr(1,nlev), qi(1,nlev,1), qir(1,nlev,1), ni(1,nlev,1), bir(1,nlev,1),       &
       th(1,nlev), th_old(1,nlev), qv_old(1,nlev), ssat(1,nlev), uz(1,nlev), ze(1,nlev),       &
       effc(1,nlev), qcl(1,nlev), dcl(1,nlev), qsn(1,nlev), dsn(1,nlev), qgr(1,nlev),         &
       dgr(1,nlev), effi(1,nlev,1), vmi(1,nlev,1), di(1,nlev,1), rhoi(1,nlev,1),               &
       d2(1,1), d3(1,nlev,ndiag), scf(1,nlev), qit(1,nlev,6), pg(1,nlev), lc(1,nlev),          &
       mt(1,nlev,11), st(1,nlev,11), mpr(1,nlev,nmicro_process_rates))

  open(12, file=trim(fout), status='replace')
  do icol = 1, ncol
     do k = 1, nlev
        read(11,*) p(1,k), rh(1,k), dz(1,k), tt(1,k), qv(1,k), qc(1,k), nc(1,k), qr(1,k), &
             nr(1,k), qi(1,k,1), qir(1,k,1), ni(1,k,1), bir(1,k,1)
     end do
     th = tt*(1.e5/p)**(rgas/cp)
     th_old = th
     qv_old = qv
     ssat = 0.; uz = 0.; d2 = 0.; d3 = 0.; mt = 0.; st = 0.; mpr = 0.; qit = 0.
     call p3_main(qc=qc, nc=nc, qr=qr, nr=nr, th_old=th_old, th=th, qv_old=qv_old, qv=qv, &
          dt=dt, qitot=qi, qirim=qir, nitot=ni, birim=bir, ssat=ssat, uzpl=uz,           &
          pres=p, dzq=dz, it=it, prt_liq=prl, prt_sol=prs, its=1, ite=1, kts=1, kte=nlev,  &
          nCat=1, diag_ze=ze, diag_effc=effc,                                               &
          diag_qice_clice=qcl, diag_Dge_clice=dcl, diag_qice_snow=qsn, diag_Dge_snow=dsn,   &
          diag_qice_graup=qgr, diag_Dge_graup=dgr,                                          &
          diag_effi=effi, diag_vmi=vmi, diag_di=di, diag_rhoi=rhoi,                         &
          n_diag_2d=1, diag_2d=d2, n_diag_3d=ndiag, diag_3d=d3,                             &
          log_predictNc=log_predictNc, typeDiags_ON=.true., model='WRF',                    &
          clbfact_dep=1., clbfact_sub=1., debug_on=.false.,                                 &
          scpf_on=.false., scpf_pfrac=0., scpf_resfact=0., SCF_out=scf,                     &
          prt_drzl=p1, prt_rain=p2, prt_crys=p3, prt_snow=p4, prt_grpl=p5, prt_pell=p6,     &
          prt_hail=p7, prt_sndp=p8, qi_type=qit, rho=rh, pgam_out=pg, lamc_out=lc,          &
          mtend=mt, stend=st, micro_proc_rates=mpr, do_accumulate_micro_proc_rates=.true.)
     tt = th*(p*1.e-5)**(rgas/cp)
     write(12,'(a,i6,2es26.17e3)') 'COL', icol, prl(1), prs(1)
     do k = 1, nlev
        write(12,'(70es26.17e3)') tt(1,k), qv(1,k), qc(1,k), nc(1,k), qr(1,k), nr(1,k), &
             qi(1,k,1), qir(1,k,1), ni(1,k,1), bir(1,k,1), mt(1,k,1:8), st(1,k,1:8),     &
             d3(1,k,1:ndiag)
     end do
  end do
  close(12)
  print*, 'P3_DRIVER_DONE', ncol, nlev
end program p3_driver
