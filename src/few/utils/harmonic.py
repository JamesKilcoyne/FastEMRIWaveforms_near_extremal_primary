from typing import Optional, Union
import numpy as np
from numba import njit
from functools import partial
from .globals import Globals, get_config

import quaternionic
import spherical
from spherical import Wigner as WignerSpherical
from spherical.wigner import WignerHindex, _complex_powers, to_euler_phases, inverse_4pi, ε

class Wigner(WignerSpherical):
    def sYlms(self, s, l, m, R, out=None, workspace=None):
        """Evaluate (possibly spin-weighted) spherical harmonic

        Parameters
        ----------
        s : int
            The spin weight of the spherical harmonic.  This must be an integer, and must satisfy `abs(s) <= self.mp_max`.
        l : array_like
            Array of non-negative integers representing the ell values for which the sYlm should be evaluated.  Each value must satisfy `ell_min <= ell <= ell_max`.
        m : array_like
            Array of integers representing the m values for which the sYlm should be evaluated.
        R : array_like
            Array to be interpreted as a quaternionic array (thus its final dimension
            must have size 4), representing the rotations on which the sYlm will be
            evaluated.
        out : array_like, optional
            Array into which the d values should be written.  It should be an array of
            complex, with size `self.Ysize`.  If not present, the array will be
            created.  In either case, the array will also be returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        Y : array
            This is a N-dimensional array of complex; see below.

        Notes
        -----
        The spherical harmonics of spin weight s are related to the 𝔇 matrix as

            ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡₘ₋ₛ(R)
                   = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇̄ˡ₋ₛₘ(R̄)

        This function is the preferred method of computing the sYlm for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        This function computes ₛYₗₘ(R).  The result is returned in a N-dimensional
        array ordered as

            [
                Y(s, ell, m, R)
                for ell, m in zip(l, m)
            ]
        
        with first dimension, N, corresponding to the dimension of R.

        """
        assert len(l) == len(m), "l and m arrays must have the same length"
        Ysize = len(l)

        if abs(s) > self.mp_max:
            raise ValueError(
                f"This object has mp_max={self.mp_max}, which is not "
                f"sufficient to compute sYlm values for spin weight s={s}"
            )
        if out is not None and out.size != (Ysize * R.size // 4):
            raise ValueError(
                f"Given output array has size {out.size}; it should be {Ysize * R.size // 4}"
            )
        if out is not None and out.dtype != complex:
            raise ValueError(f"Given output array has dtype {out.dtype}; it should be complex")

        if workspace is not None:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = self._split_workspace(workspace)
        else:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = (
                self.Hwedge, self.Hv, self.Hextra, self.zₐpowers, self.zᵧpowers, self.z
            )

        quaternions = quaternionic.array(R).ndarray.reshape((-1, 4))
        function_values = (
            out.reshape(quaternions.shape[0], Ysize)
            if out is not None
            else np.zeros(quaternions.shape[:-1] + (Ysize,), dtype=complex)
        )

        # Loop over all input quaternions
        for i_R in range(quaternions.shape[0]):
            to_euler_phases(quaternions[i_R], z)
            Hwedge = self.H(z[1], Hwedge, Hv, Hextra)
            Y = function_values[i_R]
            _complex_powers(z[0:1], self.ell_max, zₐpowers)
            zᵧpower = z[2]**abs(s)
            _fill_sYlms(self.mp_max, s, l, m, Y, Hwedge, zₐpowers[0], zᵧpower)

        return function_values.reshape(R.shape[:-1] + (Ysize,))

    def sYlms_from_spherical_coordinates(self, s, l, m, theta, phi, out=None, workspace=None):
        """Evaluate (possibly spin-weighted) spherical harmonic

        Parameters
        ----------
        s : int
            The spin weight of the spherical harmonic.  This must be an integer, and must satisfy `abs(s) <= self.mp_max`.
        l : array_like
            Array of non-negative integers representing the ell values for which the sYlm should be evaluated.  Each value must satisfy `ell_min <= ell <= ell_max`.
        m : array_like
            Array of integers representing the m values for which the sYlm should be evaluated.
        theta : array_like
            Array of polar angles (in radians) for each point at which the sYlm should be evaluated.
        phi : array_like
            Array of azimuthal angles (in radians) for each point at which the sYlm should be evaluated.
        out : array_like, optional
            Array into which the d values should be written.  It should be an array of
            complex, with size `self.Ysize`.  If not present, the array will be
            created.  In either case, the array will also be returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        Y : array
            This is a N-dimensional array of complex; see below.

        Notes
        -----
        The spherical harmonics of spin weight s are related to the 𝔇 matrix as

            ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡₘ₋ₛ(R)
                   = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇̄ˡ₋ₛₘ(R̄)

        This function is the preferred method of computing the sYlm for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        This function computes ₛYₗₘ(R).  The result is returned in a N-dimensional
        array ordered as

            [
                Y(s, ell, m, R)
                for ell, m in zip(l, m)
            ]
        
        with first dimension, N, corresponding to the dimension of R.

        """
        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        return self.sYlms(s, l, m, R, out=out, workspace=workspace)

@njit
def _fill_sYlms(mp_max, s, ell_arr, m_arr, Y, Hwedge, zₐpowers, zᵧpower):
    """Helper function for Wigner.sYlms"""
    #  ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡₘ₋ₛ(R)
    i_Y = 0
    if s >= 0:
        c1 = zᵧpower.conjugate()
        for ell, m in zip(ell_arr, m_arr):
            c2 = c1 * np.sqrt((2 * ell + 1) * inverse_4pi)
            i_H = WignerHindex(ell, m, -s, mp_max)
            if m < 0:
                Y[i_Y] = c2 * Hwedge[i_H] * zₐpowers[-m].conjugate()
            else:
                Y[i_Y] = c2 * ϵ(m) * Hwedge[i_H] * zₐpowers[m]
            i_Y += 1
    else:  # s < 0
        c1 = (-1)**s * zᵧpower
        for ell, m in zip(ell_arr, m_arr):
            c2 = c1 * np.sqrt((2 * ell + 1) * inverse_4pi)
            i_H = WignerHindex(ell, m, -s, mp_max)
            if m < 0:
                Y[i_Y] = c2 * Hwedge[i_H] * zₐpowers[-m].conjugate()
            else:
                Y[i_Y] = c2 * ϵ(m) * Hwedge[i_H] * zₐpowers[m]
            i_Y += 1

def Yslm(
    s: int, 
    l: Union[int, np.ndarray], 
    m: Union[int, np.ndarray], 
    theta: Union[float, np.ndarray], 
    phi: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
    """Evaluate spin-weighted spherical harmonic

    Parameters
    ----------
    s : int
        The spin weight of the spherical harmonic.  This must be an integer, and must satisfy `abs(s) <= self.mp_max`.
    l : int or array_like
        Non-negative integer representing the ell value for which the sYlm should be evaluated.  It must satisfy `ell_min <= ell <= ell_max`.
    m : int or array_like
        Integer representing the m value for which the sYlm should be evaluated.
    theta : float or array_like
        Polar angle (in radians) for the point at which the sYlm should be evaluated.
    phi : float or array_like
        Azimuthal angle (in radians) for the point at which the sYlm should be evaluated.   

    Returns
    -------
    Y : complex
        The value of the spin-weighted spherical harmonic at the given angles.

    """
    assert np.all(abs(s) <= l) and np.all(abs(m) <= l), f"Invalid s, l, m values: s={s}, l={l}, m={m}"
    wigner = Wigner(max(l, abs(s)), abs(s))
    if isinstance(l, int) or isinstance(m, int):
        assert isinstance(l, int) and isinstance(m, int), "l and m must both be integers or both be arrays"
        return wigner.sYlms_from_spherical_coordinates(s, np.array([l]), np.array([m]), theta, phi)[0]
    else:
        return wigner.sYlms_from_spherical_coordinates(s, l, m, theta, phi)