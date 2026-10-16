"""CAM6 ``dropmixnuc`` (ndrop.F90:290-1214): droplet activation + turbulent mixing.

Prescribed-aerosol path (``prog_modal_aero = .false.``): only the droplet-number
tendency ``tendnd`` leaves the routine (ndrop.F90:1102-1110; the aerosol-tendency
block :1114-1146 is skipped).  The droplet number depends on the aerosol only
through the per-mode NUMBER columns (interstitial ``raercol`` and cloud-borne
``raercol_cw``, via ``srcn``/``nact``), so the mass-species columns, ``mact``,
``factnum``, ``nsource``/``ndropmix``/``ndropcol`` diagnostics and the spcam gas
branch are not ported: none of them reaches ``tendnd``.  Activation uses the
interstitial aerosol from the state (``loadaer`` phase 1); the supplied
cloud-borne aerosol is zero (user decision 2026-10-16).

Mixing substeps: CAM's ``nsubmix = int(dtmicro/(0.9 dtmin)) + 1`` (:940-950) has no
cap.  JAX needs a static scan length ``n_max``, computed once at start-up by
:func:`nsubmix_upper_bound` (user decision B10); a column needing more substeps
returns NaN.

Arrays are top-first, ``(ncol, nlev)`` on layers and ``(ncol, nlev+1)`` on
interfaces; aerosol inputs carry a leading mode axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.arg_activation import (
    cam6_activate_single_updraft,
)
from legoesm.atmosphere.physics.microphysics.cam6_modal_aerosol import cam6_loadaer

__physics_contract__ = {
    "summary": "CAM6 dropmixnuc: cloud-change droplet activation/decay and explicit turbulent mixing of droplet number.",
    "inputs": {
        "ncldwtr": "1/kg (droplet number)", "temp": "K", "pmid": "Pa", "pint": "Pa (interfaces)",
        "rpdel": "1/Pa", "zm": "m (layer-centre height)", "kvh": "m^2/s (interfaces)",
        "wsub": "m/s", "cldn": "1", "cldo": "1", "cldliqf": "1", "dtmicro": "s",
        "aer_num": "1/kg (per mode)", "aer_mass": "kg/kg (per mode, per species)",
    },
    "outputs": {"tendnd": "1/kg/s (droplet-number tendency handed to the microphysics)"},
    "sign_convention": "tendnd > 0 = droplet source; qcld kept >= 0 by explmix; NaN where nsubmix > n_max",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "CAM6 ndrop.F90 dropmixnuc/explmix (Abdul-Razzak & Ghan 2000; Ghan et al. 1997)",
    "idealized_test": "gfortran build of the verbatim dropmixnuc vs this port at fp64 rtol 1e-12",
}

# --- CAM6 dropmixnuc constants (ndrop.F90:337-338, 640-650, 900-950, 1109) ----
_ZKMIN = 0.01                       # min droplet diffusivity [m^2/s]
_ZKMAX = 100.0                      # max droplet diffusivity [m^2/s]
_WMIXMIN = 0.1                      # min turbulent velocity [m/s]
_WMAX_ACT = 10.0                    # activate_modal wmaxf [m/s]
_TAU_CLD_REGENERATE_S = 3600.0 * 3.0  # cloud regeneration time scale [s]
_CLD_CHANGE_MIN = 0.01              # cloud-fraction change / presence threshold
_CLD_OVERLAP_MIN = 1.0e-10          # overlap guard
_TINV_MIN = 1.0e-6                  # dtmin only from rates above this [1/s]
_DTMIX_SAFETY = 0.9                 # dtmix = 0.9 dtmin
_QCLD_FLOOR = 1.0e-6                # tendnd uses max(qcld, 1e-6) [1/kg]
_WSUB_CAP = 10.0                    # microp_aero.F90:529-541 caps wsub at 10 m/s


def _layer_geometry(temp, pmid, pint, rpdel, zm):
    """zs, cs, dz, zn, csbot, csbot_cscen (ndrop.F90:520-552) for one sub-column."""
    g, rair = constants.g, constants.R_d
    zs_int = 1.0 / (zm[:, :-1] - zm[:, 1:])
    zs = jnp.concatenate([zs_int, zs_int[:, -1:]], axis=1)
    cs = pmid / (rair * temp)
    dz = 1.0 / (cs * g * rpdel)
    zn = g * rpdel
    csbot_int = 2.0 * pint[:, 1:-1] / (rair * (temp[:, :-1] + temp[:, 1:]))
    csbot = jnp.concatenate([csbot_int, cs[:, -1:]], axis=1)
    csbot_cscen = jnp.concatenate([csbot_int / cs[:, :-1], jnp.ones_like(cs[:, -1:])], axis=1)
    return zs, cs, dz, zn, csbot, csbot_cscen


def _shift_down(q):
    """q(kp1) with kp1 = min(k+1, pver)."""
    return jnp.concatenate([q[:, 1:], q[:, -1:]], axis=1)


def _shift_up(q):
    """q(km1) with km1 = max(k-1, top_lev)."""
    return jnp.concatenate([q[:, :1], q[:, :-1]], axis=1)


def _explmix_act(qold, src, ekkp, ekkm, overlapp, overlapm, dt):
    """explmix, is_unact = .false., surfrate = flxconv = 0 (ndrop.F90:1270-1286)."""
    q = qold + dt * (src + ekkp * (overlapp * _shift_down(qold) - qold)
                     + ekkm * (overlapm * _shift_up(qold) - qold))
    return jnp.maximum(q, 0.0)


def _explmix_unact(qold, src, ekkp, ekkm, overlapp, overlapm, dt, qactold):
    """explmix, is_unact = .true., surfrate = flxconv = 0 (ndrop.F90:1245-1268)."""
    q = qold + dt * (-src + ekkp * (_shift_down(qold) - qold
                                    + _shift_down(qactold) * (1.0 - overlapp))
                     + ekkm * (_shift_up(qold) - qold
                               + _shift_up(qactold) * (1.0 - overlapm)))
    return jnp.maximum(q, 0.0)


def _ekkp_ekkm(ekd, csbot, zn, zs):
    """ekk(k<pver) = ekd*csbot; ekk(top-1) = ekk(pver) = 0; ekkp/ekkm (:902-922)."""
    ekk = (ekd * csbot).at[:, -1].set(0.0)            # ekk(k), k = top..pver
    ekk_above = jnp.concatenate([jnp.zeros_like(ekk[:, :1]), ekk[:, :-1]], axis=1)  # ekk(k-1)
    ekkp = zn * ekk * zs
    ekkm = zn * ekk_above * _shift_up(zs)
    return ekkp, ekkm


def _nsubmix(tinv, dtmicro):
    dtmin = jnp.min(jnp.where(tinv > _TINV_MIN, 1.0 / jnp.where(tinv > _TINV_MIN, tinv, 1.0),
                              dtmicro), axis=1)
    dtmin = jnp.minimum(dtmin, dtmicro)
    # Fortran: nsubmix = dtmicro/dtmix + 1 (real sum, then truncation toward zero)
    nsubmix = jnp.floor(dtmicro / (_DTMIX_SAFETY * dtmin) + 1.0)
    return nsubmix, dtmicro / nsubmix


def nsubmix_upper_bound(temp, pmid, pint, rpdel, zm, dtmicro, top_lev):
    """Largest ``nsubmix`` any column can need, from CAM's dtmin formula with every
    rate at its maximum: ``ekd <= max(zkmax, wsub_cap/zs)`` (cloud-base layers set
    ``ekd = wbar/zs``, ndrop.F90:762) and ``taumix_internal_pver_inv <= wsub_cap/dz``
    (:848).  Evaluated on the state passed (the start-up state, user decision B10)."""
    sl = slice(top_lev, None)
    zs, cs, dz, zn, csbot, _ = _layer_geometry(temp[:, sl], pmid[:, sl], pint[:, top_lev:],
                                               rpdel[:, sl], zm[:, sl])
    ekd = jnp.maximum(_ZKMAX, _WSUB_CAP / zs)
    ekkp, ekkm = _ekkp_ekkm(ekd, csbot, zn, zs)
    tinv = (ekkp + ekkm).at[:, -1].add(_WSUB_CAP / dz[:, -1])
    nsub, _ = _nsubmix(tinv, dtmicro)
    return int(jnp.max(nsub))


def cam6_dropmixnuc(ncldwtr, temp, pmid, pint, rpdel, zm, kvh, wsub, cldn, cldo, cldliqf,
                    dtmicro, aer_num, aer_mass, props, top_lev, n_max):
    """Droplet-number tendency ``tendnd`` [1/kg/s] (zero above ``top_lev``).

    ``aer_num`` (nmode, ncol, nlev) [1/kg]; ``aer_mass`` tuple of (nspec_m, ncol, nlev)
    [kg/kg] in MAM4_SPECIES order; ``props`` a ``Cam6ModeProps``; ``top_lev`` the
    0-based first cloud level (static); ``n_max`` the static substep bound.
    """
    nlev_full = temp.shape[1]
    sl = slice(top_lev, None)
    temp, pmid, rpdel, zm = temp[:, sl], pmid[:, sl], rpdel[:, sl], zm[:, sl]
    ncldwtr, wsub = ncldwtr[:, sl], wsub[:, sl]
    cldn, cldo, cldliqf = cldn[:, sl], cldo[:, sl], cldliqf[:, sl]
    pint, kvh = pint[:, top_lev:], kvh[:, top_lev:]
    aer_num = aer_num[:, :, sl]
    aer_mass = tuple(a[:, :, sl] for a in aer_mass)
    nmode = aer_num.shape[0]
    dtinv = 1.0 / dtmicro

    zs, cs, dz, zn, csbot, csbot_cscen = _layer_geometry(temp, pmid, pint, rpdel, zm)
    ekd = jnp.minimum(jnp.maximum(kvh[:, 1:-1], _ZKMIN), _ZKMAX)
    ekd = jnp.concatenate([ekd, jnp.zeros_like(ekd[:, :1])], axis=1)
    wtke = jnp.maximum(wsub, _WMIXMIN)
    lcldo = cldo * cldliqf
    lcldn = cldn * cldliqf

    # aerosol per mode at k (state) -> activation inputs
    def load(m, mass, num, rho):
        return cam6_loadaer(mass, num, rho, props.specdens[m], props.spechygro[m],
                            props.voltonumblo[m], props.voltonumbhi[m])
    acts = [load(m, aer_mass[m], aer_num[m], cs) for m in range(nmode)]
    # old-cloud activation reads the aerosol of layer kp1 = min(k+1, pver), with cs(kp1)
    acts_kp1 = [load(m, jnp.stack([_shift_down(s) for s in aer_mass[m]]), _shift_down(aer_num[m]),
                     _shift_down(cs)) for m in range(nmode)]

    qcld = ncldwtr
    R = [aer_num[m] for m in range(nmode)]            # raercol(number, nsav)
    C = [jnp.zeros_like(aer_num[m]) for m in range(nmode)]  # raercol_cw(number): supplied cw = 0

    # --- grow_shrink_main_k_loop (:640-735) ---------------------------------
    cldo_i = cldo * (1.0 - cldliqf)
    cldn_i = cldn * (1.0 - cldliqf)
    shrink_i = cldn_i < cldo_i
    dumc = jnp.where(shrink_i, (cldn_i - cldo_i) / jnp.where(shrink_i, cldo_i, 1.0)
                     * (1.0 - cldliqf), 0.0)
    for m in range(nmode):
        dact = C[m] * dumc
        C[m], R[m] = C[m] + dact, R[m] - dact
    cldo_tmp = lcldo
    cldn_tmp = lcldn * jnp.exp(-dtmicro / _TAU_CLD_REGENERATE_S)
    shrink = cldn_tmp < cldo_tmp
    frac = jnp.where(shrink, (cldn_tmp - cldo_tmp) / jnp.where(shrink, cldo_tmp, 1.0), 0.0)
    qcld = jnp.where(shrink, qcld * (1.0 + frac), qcld)
    dumc = frac * cldliqf
    for m in range(nmode):
        dact = C[m] * dumc
        C[m], R[m] = C[m] + dact, R[m] - dact
    grow = lcldn - cldn_tmp > _CLD_CHANGE_MIN
    na = jnp.stack([a[0] for a in acts]); va = jnp.stack([a[1] for a in acts])
    hy = jnp.stack([a[2] for a in acts])
    fn_g = cam6_activate_single_updraft(wtke, temp, cs, na, va, hy, props.sigmag, _WMAX_ACT)[0]
    dumc = jnp.where(grow, lcldn - cldn_tmp, 0.0)
    for m in range(nmode):
        dact = dumc * fn_g[m] * aer_num[m]
        qcld = qcld + dact
        C[m], R[m] = C[m] + dact, R[m] - dact

    # --- old_cloud_main_k_loop (:737-892) -----------------------------------
    lcldn_kp1 = _shift_down(lcldn)
    incloud = lcldn > _CLD_CHANGE_MIN
    is_bot = jnp.zeros_like(lcldn, dtype=bool).at[:, -1].set(True)
    base = incloud & ((lcldn - lcldn_kp1 > _CLD_CHANGE_MIN) | is_bot)
    ekd = jnp.where(base, wtke / zs, ekd)
    na1 = jnp.stack([a[0] for a in acts_kp1]); va1 = jnp.stack([a[1] for a in acts_kp1])
    hy1 = jnp.stack([a[2] for a in acts_kp1])
    _, _, fluxn, _, ffa = cam6_activate_single_updraft(wtke, temp, cs, na1, va1, hy1,
                                                       props.sigmag, _WMAX_ACT)
    dumc = jnp.where(is_bot, lcldn, lcldn - lcldn_kp1)
    dum = csbot_cscen / dz
    nact = [jnp.where(base, fluxn[m] * dumc * dum, 0.0) for m in range(nmode)]
    taumix = jnp.where(base[:, -1], ffa[:, -1] / dz[:, -1], 0.0)        # (ncol,)
    clear = ~incloud
    qcld = jnp.where(clear, 0.0, qcld)
    merge = clear & (cldn < _CLD_CHANGE_MIN)
    for m in range(nmode):
        R[m] = jnp.where(merge, R[m] + C[m], R[m])
        C[m] = jnp.where(merge, 0.0, C[m])

    # --- substep setup (:898-978) -------------------------------------------
    ekkp, ekkm = _ekkp_ekkm(ekd, csbot, zn, zs)
    tinv = (ekkp + ekkm).at[:, -1].add(taumix)
    nsubmix, dtmix = _nsubmix(tinv, dtmicro)
    cldn_kp1, cldn_km1 = _shift_down(cldn), _shift_up(cldn)
    overlapp = jnp.where(cldn_kp1 > _CLD_OVERLAP_MIN,
                         jnp.minimum(cldn / jnp.where(cldn_kp1 > _CLD_OVERLAP_MIN, cldn_kp1, 1.0), 1.0), 1.0)
    overlapm = jnp.where(cldn_km1 > _CLD_OVERLAP_MIN,
                         jnp.minimum(cldn / jnp.where(cldn_km1 > _CLD_OVERLAP_MIN, cldn_km1, 1.0), 1.0), 1.0)
    nact = [jnp.concatenate([jnp.minimum(a[:, :-1], ekkp[:, :-1]), a[:, -1:]], axis=1) for a in nact]

    def src_of(Rm, Cm, nactm):
        top = nactm[:, :-1] * Rm[:, 1:]
        bot = jnp.maximum(0.0, Rm[:, -1] * nactm[:, -1] + Cm[:, -1] * (nactm[:, -1] - taumix))
        return jnp.concatenate([top, bot[:, None]], axis=1)

    dt_col = dtmix[:, None]

    def substep(carry, n):
        qcld, R, C = carry
        live = (n < nsubmix)[:, None]
        srcs = [src_of(R[m], C[m], nact[m]) for m in range(nmode)]
        srcn = srcs[0]
        for s in srcs[1:]:
            srcn = srcn + s
        q_new = _explmix_act(qcld, srcn, ekkp, ekkm, overlapp, overlapm, dt_col)
        R_new, C_new = [], []
        for m in range(nmode):
            C_new.append(_explmix_act(C[m], srcs[m], ekkp, ekkm, overlapp, overlapm, dt_col))
            R_new.append(_explmix_unact(R[m], srcs[m], ekkp, ekkm, overlapp, overlapm, dt_col, C[m]))
        pick = lambda a, b: jnp.where(live, a, b)
        return (pick(q_new, qcld), [pick(a, b) for a, b in zip(R_new, R)],
                [pick(a, b) for a, b in zip(C_new, C)]), None

    (qcld, R, C), _ = jax.lax.scan(substep, (qcld, R, C), jnp.arange(n_max, dtype=temp.dtype))
    qcld = jnp.where(cldn == 0.0, 0.0, qcld)
    tendnd = (jnp.maximum(qcld, _QCLD_FLOOR) - ncldwtr) * dtinv
    tendnd = jnp.where((nsubmix > n_max)[:, None], jnp.nan, tendnd)
    pad = jnp.zeros((tendnd.shape[0], top_lev), dtype=tendnd.dtype)
    return jnp.concatenate([pad, tendnd], axis=1)[:, :nlev_full]
