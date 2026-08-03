from math import log
from typing import Optional, Union

import numpy as np
from numba import njit

from ..geodesic import get_separatrix

XMIN = 0.05
AMAX = 0.999
AMIN = -AMAX
DELTAPMIN = 0.001
DELTAPMAX = 9 + DELTAPMIN
EMAX = 0.9
ESEP = 0.25

ALPHA_FLUX = 1.0 / 2.0
BETA_FLUX = 2.0
ALPHA_AMP = 1.0 / 3.0
BETA_AMP = 3.0

DPC_REGIONB = DELTAPMAX - 0.001
PMAX_REGIONB = 200
AMAX_REGIONB = 0.999
AMIN_REGIONB = -AMAX_REGIONB
EMAX_REGIONB = 0.9
DELTAPMIN_REGIONB = 9

#Parameters for region C (near-extremal region)

DELTAPMIN_REGIONC = 10**(-1.625)           
DELTAPMIN_REGIONC_amp= 10**(-2)
DELTAPMAX_nex = 10
DELTAPMAX_REGIONC = DELTAPMAX_nex     #redundant
PMAX_REGIONC = 11

AMAX_nex =  1-10**(-12)# AMAX   # 1-10**(-12)
AMIN_nex = 0.99  # AMAX  
EMAX_nex =  0.4
ALPHA_FLUX_nex= 1/4#ALPHA_FLUX
BETA_FLUX_nex = BETA_FLUX

ALPHA_AMP_nex= 1/4#ALPHA_FLUX
BETA_AMP_nex = BETA_AMP


@njit
def _kerrecceq_flux_forward_map(
    a: float,
    p: float,
    e: float,
    xI: float,
    pLSO: float,
):
    """
    A jit-compiled forward mapping for the flux grids, optimised for fast ODE evaluations.
    """
    if p <= pLSO + DELTAPMAX:
        return *_uwyz_of_apex_kernel(a, p, e, xI, pLSO, ALPHA_FLUX, BETA_FLUX), True
    else:
        return *_UWYZ_of_apex_kernel(
            a,
            p,
            e,
            xI,
            pLSO,
            True,
        ), False


def kerrecceq_forward_map(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    xI: Union[float, np.ndarray],
    pLSO: Optional[Union[float, np.ndarray]] = None,
    return_mask: bool = False,
    kind: str = "flux",
):
    """
    Map from apex coordinates to interpolation coordinates for the KerrEccEq model.
    This mapping returns the interpolation coordinates (u, w, y, z) corresponding to the apex coordinates (p, e, xI, a).

    The interpolation coordinates are defined on the range [0, 1] and refer to either a "close" region A or a "far" region B.
    To also output an array indicating whether each point is in region A or B, set return_mask=True.

    For either flux or amplitude interpolation with the KerrEccEq model, the same formulae are applied with different tunable
    parameters. The default is flux interpolation, but amplitude interpolation can be selected by setting kind="amplitude".

    Arguments:
        a (float or np.ndarray): Spin parameter of the massive black hole.
        p (float or np.ndarray): Semi-latus rectum of inspiral.
        e (float or np.ndarray): Eccentricity of inspiral.
        xI (float or np.ndarray): Cosine of inclination of inspiral. Only a value of 1 is supported.
        pLSO (float or np.ndarray, optional): Separatrix value for the given parameters. If not provided, it will be computed.
        return_mask (bool, optional): If True, return a mask indicating whether the point is in region A or B.
            A value of True corresponds to region A. For kind="flux", a mask is always returned for efficiency.
            Default is False.
        kind (str, optional): Type of mapping to perform. Default is "flux".
    """
    xp = np  # TODO: gpu

    if np.any(xI != 1):
        raise ValueError("Only xI = 1 is supported.")

    if kind == "flux":
        is_flux = True
        alpha = ALPHA_FLUX
        beta = BETA_FLUX
    elif kind == "amplitude":
        is_flux = False
        alpha = ALPHA_AMP
        beta = BETA_AMP
    else:
        raise ValueError

    a = xp.atleast_1d(xp.asarray(a))
    p = xp.atleast_1d(xp.asarray(p))
    e = xp.atleast_1d(xp.asarray(e))
    xI = xp.atleast_1d(xp.asarray(xI))

    u = xp.zeros_like(a)
    w = xp.zeros_like(a)
    y = xp.zeros_like(a)
    z = xp.zeros_like(a)

    a_sep_in = xp.abs(a)

    asign = xp.sign(a)
    asign[asign == 0] = 1

    xI_sep_in = asign * xI

    # compute separatrix at all points
    if pLSO is None:
        pLSO = get_separatrix(a_sep_in, e, xI_sep_in)
    else:
        pLSO = xp.atleast_1d(xp.asarray(pLSO))
        if pLSO.shape != a.shape:
            raise ValueError("pLSO must have the same shape as a, p, e, and xI.")
        if xp.any(pLSO <= 0):
            raise ValueError("All values of pLSO must be positive.")

    # handle regions A and B
    near = p <= pLSO + DELTAPMAX
    if xp.any(near):
        out = _uwyz_of_apex_kernel(
            a[near], p[near], e[near], xI[near], pLSO[near], alpha, beta
        )
        u[near] = out[0]
        w[near] = out[1]
        y[near] = out[2]
        z[near] = out[3]

    far = np.bitwise_not(near)
    if xp.any(far):
        out = _UWYZ_of_apex_kernel(a[far], p[far], e[far], xI[far], pLSO[far], is_flux)
        u[far] = out[0]
        w[far] = out[1]
        y[far] = out[2]
        z[far] = out[3]

    for coord in [u, w, z]:
        near_one_mask = xp.abs(coord - 1) < 1e-12
        near_zero_mask = xp.abs(coord) < 1e-12
        coord[near_one_mask] = 1.0
        coord[near_zero_mask] = 0.0

    if return_mask:
        return u, w, y, z, near
    else:
        return u, w, y, z


def kerrecceq_backward_map(
    u: Union[float, np.ndarray],
    w: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    regionA: bool = True,
    kind: str = "flux",
):
    """
    Map from interpolation coordinates to apex coordinates for the KerrEccEq model.
    This mapping returns the apex coordinates (a, p, e, xI) corresponding to the interpolation coordinates (u, w, y, z).

    There are two sets of interpolation coordinates. Both are defined on the range [0, 1], with one valid in a "close" region A
    and the other in a "far" region B.

    For either flux or amplitude interpolation with the KerrEccEq model, the same formulae are applied with different tunable
    parameters. The default is flux interpolation, but amplitude interpolation can be selected by setting kind="amplitude".

    Arguments:
        u (float or np.ndarray): Interpolation coordinate corresponding to the semi-latus rectum of inspiral.
        w (float or np.ndarray): Interpolation coordinate corresponding to the eccentricity of inspiral.
        y (float or np.ndarray): Interpolation coordinate corresponding to the inclination of inspiral.
        z (float or np.ndarray): Interpolation coordinate corresponding to the spin of the massive black hole.
        regionA (bool, optional): If True, perform the transformation in RegionA. If False, perform transformation in RegionB.
            Default is True.
        kind (str, optional): Type of mapping to perform. Default is "flux".
    """
    xp = np  # TODO: gpu

    if np.any(np.asarray(y) != 1):
        raise ValueError("Only xI = 1 is supported.")

    is_flux = kind == "flux"
    is_amp = kind == "amplitude"
    if is_flux:
        alpha = ALPHA_FLUX
        beta = BETA_FLUX
    elif is_amp:
        alpha = ALPHA_AMP
        beta = BETA_AMP
    else:
        raise ValueError

    # if scalar directly evaluate the kernel for speed
    if not hasattr(u, "__len__"):
        if regionA:
            return apex_of_uwyz(u, w, y, z, alpha=alpha, beta=beta)
        else:
            return apex_of_UWYZ(u, w, y, z, is_flux)

    # else, we have multiple points
    u = xp.atleast_1d(xp.asarray(u))
    w = xp.atleast_1d(xp.asarray(w))
    y = xp.atleast_1d(xp.asarray(y))
    z = xp.atleast_1d(xp.asarray(z))

    if regionA:
        return apex_of_uwyz(u, w, y, z, alpha=alpha, beta=beta)
    else:
        return apex_of_UWYZ(u, w, y, z, is_flux)


@njit
def u_of_p(p, pLSO, alpha):
    """
    See Eq. (B7a) of https://doi.org/10.48550/arXiv.2506.09470
    """
    check_term = (
        np.log(p - pLSO + DELTAPMAX - 2 * DELTAPMIN) - log(DELTAPMAX - DELTAPMIN)
    ) / log(2)
    sgn = np.sign(check_term)
    return sgn * (sgn * check_term) ** alpha


@njit
def u_of_p_flux(p, pLSO):
    """
    u_of_p for flux interpolation, with alpha = ALPHA_FLUX. See Eq. (B7a) of https://doi.org/10.48550/arXiv.2506.09470
    """
    check_term = (
        np.log(p - pLSO + DELTAPMAX - 2 * DELTAPMIN) - log(DELTAPMAX - DELTAPMIN)
    ) / log(2)
    if check_term < 0:
        return -((-check_term) ** ALPHA_FLUX)
    else:
        return check_term**ALPHA_FLUX


@njit
def y_of_x(x):
    """
    See Eq. (B7c) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return (x - XMIN) / (1 - XMIN)


@njit
def chi_of_a(a):
    """
    See Eq. (B7)-(B8) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return (1 - a) ** (1 / 3)


@njit
def z_of_a(a):
    """
    See Eq. (B7d) of https://doi.org/10.48550/arXiv.2506.09470
    """
    chimax = chi_of_a(AMIN)
    chimin = chi_of_a(AMAX)
    return (chi_of_a(a) - chimin) / (chimax - chimin)


@njit
def Secc_of_uz(
    u,
    z,
    beta,
):
    """
    See Eq. (B8) of https://doi.org/10.48550/arXiv.2506.09470
    """
    check_part = z + u**beta * (1 - z)
    sgn = np.sign(check_part)
    return ESEP + (EMAX - ESEP) * sgn * np.sqrt(sgn * check_part)


@njit
def w_of_euz(e, u, z, beta):
    """
    See Eq. (B7b) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return e / Secc_of_uz(u, z, beta)


@njit
def w_of_euz_flux(e, u, z):
    """
    w_of_euz for flux interpolation, with beta = BETA_FLUX. See Eq. (B7b) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return e / Secc_of_uz(u, z, BETA_FLUX)


@njit
def p_of_u(u, pLSO, alpha):
    """
    Inverse of u_of_p. See Eq. (B7a) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (
        np.exp(u ** (1 / alpha) * log(2)) - 1
    )


@njit
def p_of_u_flux(u, pLSO):
    """
    Inverse of u_of_p_flux. See Eq. (B7a) of https://doi.org/10.48550/arXiv.2506.09470 with alpha = ALPHA_FLUX
    """
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (
        np.exp(u ** (1 / ALPHA_FLUX) * log(2)) - 1
    )


@njit
def x_of_y(y):
    """
    Inverse of y_of_x. See Eq. (B7c) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return y * (1 - XMIN) + XMIN

@njit
def a_of_chi(chi):
    """
    Inverse of chi_of_a. See Eq. (B7)-(B8) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return 1 - chi**3


@njit
def a_of_z(z):
    """
    Inverse of z_of_a. See Eq. (B7d) of https://doi.org/10.48550/arXiv.2506.09470
    """
    chimax = chi_of_a(AMIN)
    chimin = chi_of_a(AMAX)
    return a_of_chi(chimin + z * (chimax - chimin))


@njit
def e_of_uwz(u, w, z, beta):
    """
    Inverse of w_of_euz. See Eq. (B7b) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return Secc_of_uz(u, z, beta) * w


@njit
def e_of_uwz_flux(u, w, z):
    """
    Inverse of w_of_euz_flux. See Eq. (B7b) of https://doi.org/10.48550/arXiv.2506.09470 with beta = BETA_FLUX
    """
    return Secc_of_uz(u, z, BETA_FLUX) * w


@njit
def u_where_w_is_unity(e, z, kind="flux"):
    """
    Utility function for determining u value where w = 1, which corresponds to e = Secc. This is used in the backward mapping to determine whether a 
    point is in region A or B based on the value of w, since w = 1 corresponds to the boundary between the two regions. See Eq. (B7b) of https://doi.org/10.48550/arXiv.2506.09470
    """
    if kind == "flux":
        beta = BETA_FLUX
    elif kind == "amplitude":
        beta = BETA_AMP

    if z == 1:
        return 0.0

    if e < ESEP:
        return np.nan

    part = ((e - ESEP) / (EMAX - ESEP)) ** 2
    inside_root = (part - z) / (1 - z)
    return inside_root ** (1 / beta)


@njit
def _uwyz_of_apex_kernel(
    a,
    p,
    e,
    x,
    pLSO,
    alpha,
    beta,
):
    u = u_of_p(p, pLSO, alpha)
    y = y_of_x(x)
    z = z_of_a(a)
    w = w_of_euz(e, u, z, beta)
    return u, w, y, z


def apex_of_uwyz(
    u,
    w,
    y,
    z,
    alpha=ALPHA_FLUX,
    beta=BETA_FLUX,
):
    a = a_of_z(z)
    x = x_of_y(y)
    e = e_of_uwz(
        u,
        w,
        z,
        beta,
    )
    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    p = p_of_u(u, pLSO, alpha)
    return a, p, e, x


# # Region B


@njit
def U_of_p_flux(p, pLSO):
    """
    See Eq. (B14) of https://doi.org/10.48550/arXiv.2506.09470 with 
    """
    return ((DELTAPMIN_REGIONB) ** (-0.5) - (p - pLSO) ** (-0.5)) / (
        (DELTAPMIN_REGIONB) ** (-0.5) - (PMAX_REGIONB - pLSO) ** (-0.5)
    )


@njit
def p_of_U_flux(U, pLSO):
    """
    Inverse of U_of_p_flux. See Eq. (B14) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return (
        (DELTAPMIN_REGIONB) ** (-0.5)
        - U * ((DELTAPMIN_REGIONB) ** (-0.5) - (PMAX_REGIONB - pLSO) ** (-0.5))
    ) ** (-2) + pLSO


@njit
def U_of_p_amplitude(p, pLSO):
    """
    See Eq. (B19) of https://doi.org/10.48550/arXiv.2506.09470
    """
    pc = pLSO + DPC_REGIONB
    return (pc ** (-0.5) - p ** (-0.5)) / (pc ** (-0.5) - (PMAX_REGIONB + pc) ** (-0.5))


@njit
def p_of_U_amplitude(U, pLSO):
    """
    Inverse of U_of_p_amplitude. See Eq. (B19) of https://doi.org/10.48550/arXiv.2506.09470
    """
    pc = pLSO + DPC_REGIONB
    return (pc ** (-0.5) - U * (pc ** (-0.5) - (PMAX_REGIONB + pc) ** (-0.5))) ** (-2)


@njit
def W_of_e(e):
    """
    See Eqs. (B15) and (B20) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return e / EMAX_REGIONB


@njit
def e_of_W(w):
    return w * EMAX_REGIONB


@njit
def Y_of_x(x):
    """
    See Eqs. (B16) and (B21) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return y_of_x(x)


@njit
def x_of_Y(y):
    """
    Inverse of Y_of_x. See Eqs. (B16) and (B21) of https://doi.org/10.48550/arXiv.2506.09470
    """
    return x_of_y(y)


@njit
def Z_of_a(a):
    """
    See Eqs. (B17) and (B22) of https://doi.org/10.48550/arXiv.2506.09470
    """
    chimax = chi_of_a(AMIN_REGIONB)
    chimin = chi_of_a(AMAX_REGIONB)
    return (chi_of_a(a) - chimin) / (chimax - chimin)


@njit
def a_of_Z(z):
    """
    Inverse of Z_of_a. See Eqs. (B17) and (B22) of https://doi.org/10.48550/arXiv.2506.09470
    """
    chimax = chi_of_a(AMIN_REGIONB)
    chimin = chi_of_a(AMAX_REGIONB)
    return a_of_chi(chimin + z * (chimax - chimin))


@njit
def _UWYZ_of_apex_kernel(
    a,
    p,
    e,
    x,
    pLSO,
    is_flux,
):
    if is_flux:
        u = U_of_p_flux(p, pLSO)
    else:
        u = U_of_p_amplitude(p, pLSO)
    y = Y_of_x(x)
    z = Z_of_a(a)
    w = W_of_e(e)
    return u, w, y, z


def apex_of_UWYZ(
    u,
    w,
    y,
    z,
    is_flux,
):
    a = a_of_Z(z)
    x = x_of_Y(y)
    e = e_of_W(w)

    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    if is_flux:
        p = p_of_U_flux(u, pLSO)
    else:
        p = p_of_U_amplitude(u, pLSO)
    return a, p, e, x

   # # RegionC
#Keep (p,u) the same (different bounds), change (a,z), keep (x,y),  alter (e,w)



#(p,u)

@njit
def u_of_p_flux_nex(p,pLSO):
    num = np.log10(p-pLSO) - np.log10(DELTAPMIN_REGIONC)
    den = np.log10(DELTAPMAX_REGIONC) - np.log10(DELTAPMIN_REGIONC)
    return num/den

@njit
def u_of_p_nex(p,pLSO, deltapmin):
    num = np.log10(p-pLSO) - np.log10(deltapmin)
    den = np.log10(DELTAPMAX_REGIONC) - np.log10(deltapmin)
    return num/den


@njit
def p_of_u_nex(u,pLSO, deltapmin):
    return 10**(u*(np.log10(DELTAPMAX_REGIONC)-np.log10(deltapmin)) + np.log10(deltapmin))+pLSO

@njit
def p_of_u_flux_nex(u,pLSO):
    return 10**(u*(np.log10(DELTAPMAX_REGIONC)-np.log10(DELTAPMIN_REGIONC)) + np.log10(DELTAPMIN_REGIONC))+pLSO

#(a,z)

@njit
def chi_of_a_nex(a):
    return np.log10(1-a)

@njit
def z_of_a_nex(a):
    chimax = chi_of_a_nex(AMIN_nex)
    chimin = chi_of_a_nex(AMAX_nex)
    return (chi_of_a_nex(a) - chimax) / (chimin - chimax)


@njit
def a_of_chi_nex(chi):
    return 1 - 10**(chi)


@njit
def a_of_z_nex(z):
    chimax = chi_of_a_nex(AMIN_nex)
    chimin = chi_of_a_nex(AMAX_nex)
    return a_of_chi_nex(chimax + z * (chimin - chimax))


#(e,w)

@njit
def w_of_e_nex(e):
    return e / EMAX_nex


@njit
def w_of_e_flux_nex(e):
    return e / EMAX_nex

@njit
def e_of_w_nex(w):
    return  w*EMAX_nex

@njit
def e_of_w_flux_nex(w):
    return w*EMAX_nex


@njit
def _uwyz_of_apex_kernel_nex(
    a,
    p,
    e,
    x,
    pLSO,
    deltapmin
):
    u = u_of_p_nex(p, pLSO, deltapmin)
    y = y_of_x(x)
    z = z_of_a_nex(a)
    w = w_of_e_nex(e)
    return u, w, y, z


def apex_of_uwyz_nex(
    u,
    w,
    y,
    z,
    deltapmin
):
    a = a_of_z_nex(z)
    x = x_of_y(y)
    e = e_of_w_nex(w)
    a = np.asarray(a)
    a_in = np.abs(a)
    x_in = np.sign(a)
    x_in[x_in == 0] = 1

    pLSO = get_separatrix(a_in, e, x_in)
    p = p_of_u_nex(u, pLSO,deltapmin)

    return a, p, e, x

@njit
def _kerrecceq_flux_forward_map_nex(
    a: float,
    p: float,
    e: float,
    xI: float,
    pLSO: float,
):
    """
    A jit-compiled forward mapping for the flux grids, optimised for fast ODE evaluations.
    """


    return *_uwyz_of_apex_kernel_nex(a, p, e, xI, pLSO,DELTAPMIN_REGIONC), "RegionC"


def kerrecceq_forward_map_nex(
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    xI: Union[float, np.ndarray],
    pLSO: Optional[Union[float, np.ndarray]] = None,
    return_mask: bool = False,
    kind: str = "flux",
):


    xp = np  # TODO: gpu

    if np.any(xI != 1):
        raise ValueError("Only xI = 1 is supported.")

    if kind == "flux":
        is_flux = True
        alpha = ALPHA_FLUX_nex
        beta = BETA_FLUX_nex
        deltapmin = DELTAPMIN_REGIONC
    elif kind == "amplitude":
        is_flux = False
        alpha = ALPHA_AMP_nex
        beta = BETA_AMP_nex
        deltapmin = DELTAPMIN_REGIONC_amp
    else:
        raise ValueError

    a = xp.atleast_1d(xp.asarray(a))
    p = xp.atleast_1d(xp.asarray(p))
    e = xp.atleast_1d(xp.asarray(e))
    xI = xp.atleast_1d(xp.asarray(xI))

    u = xp.zeros_like(a)
    w = xp.zeros_like(a)
    y = xp.zeros_like(a)
    z = xp.zeros_like(a)

    a_sep_in = xp.abs(a)

    asign = xp.sign(a)
    asign[asign == 0] = 1

    xI_sep_in = asign * xI

    # compute separatrix at all points
    pLSO = get_separatrix(a_sep_in, e, xI_sep_in)

    # handle regions A and B and C

    near = (p <= (pLSO + DELTAPMAX)*0) & (a< AMAX_nex)        #change near and far so that they are enver true, quickfix
    far = (p> (pLSO+DELTAPMAX)*1000) & (p<=PMAX_REGIONB)
    near_extremal = (p <= pLSO + DELTAPMAX_nex)  & (a>=AMIN_nex)


    if xp.any(near):
        raise ValueError

    if xp.any(far):
        raise ValueError


    if xp.any(near_extremal):
        out = _uwyz_of_apex_kernel_nex(a[near_extremal], p[near_extremal], e[near_extremal], xI[near_extremal], pLSO[near_extremal],deltapmin) 
        u[near_extremal] = out[0]
        w[near_extremal] = out[1]
        y[near_extremal] = out[2]
        z[near_extremal] = out[3]

    if return_mask:
        return u, w, y, z, near_extremal
    else:
        return u, w, y, z
    



def kerrecceq_backward_map_nex(
    u: Union[float, np.ndarray],
    w: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    Region: str = "RegionC",
    kind: str = "flux",
):
    if np.any(np.asarray(y) != 1):
        raise ValueError("Only xI = 1 is supported.")

    xp= np
    
    is_flux = kind == "flux"
    is_amp = kind == "amplitude"
    if is_flux:
        alpha = ALPHA_FLUX_nex
        beta = BETA_FLUX_nex
        deltapmin = DELTAPMIN_REGIONC
    elif is_amp:
        alpha = ALPHA_AMP_nex
        beta = BETA_AMP_nex
        deltapmin = DELTAPMIN_REGIONC_amp
    else:
        raise ValueError
    
    # if scalar directly evaluate the kernel for speed
    if not hasattr(u, "__len__"):
        return apex_of_uwyz_nex(u, w, y, z,deltapmin)
    
    # else, we have multiple points
    u = xp.atleast_1d(xp.asarray(u))
    w = xp.atleast_1d(xp.asarray(w))
    y = xp.atleast_1d(xp.asarray(y))
    z = xp.atleast_1d(xp.asarray(z))
    
    return apex_of_uwyz_nex(u, w, y, z,deltapmin)