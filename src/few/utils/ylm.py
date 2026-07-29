# Function for ylm generation for FastEMRIWaveforms Packages

from typing import Optional, Union

import numpy as np
from .harmonic import Wigner

# base classes
from ..utils.baseclasses import ParallelModuleBase

I = 1j

class GetYlms(ParallelModuleBase):
    r"""Spin-weighted Spherical Harmonics

    The class generates spin-weighted spherical harmonics,
    :math:`Y_{lm}(\Theta,\phi)`.

    args:
        spin_weight: The spin weight of the spherical harmonics. Default is -2.
        include_minus_m: Set True if only providing :math:`m\geq0`,
            it will return twice the number of requested modes with the second
            half as modes with :math:`m<0` for array inputs of :math:`l,m`. **Warning**: It will also duplicate
            the :math:`m=0` modes. Default is False.
        lmax: The maximum :math:`l` value to support. Default is 30. Note that the number of modes scales as :math:`\ell_\mathrm{max}^2`, so setting this too high may lead to long runtimes and large memory usage.
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
    """

    def __init__(self, spin_weight: int = -2, include_minus_m: bool = False, lmax: int = 18, **kwargs: Optional[dict]):
        ParallelModuleBase.__init__(self, **kwargs)
        self.spin_weight = spin_weight
        self.include_minus_m = include_minus_m
        self.lmax = lmax
        self.lmin = max(abs(spin_weight), 0)
        self.wigner = Wigner(lmax, self.lmin)

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def __call__(
        self,
        l_in: Union[int, np.ndarray],
        m_in: Union[int, np.ndarray],
        theta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray],
        include_minus_m: Optional[bool] = None,
        bounds_check: bool = True
    ) -> np.ndarray:
        """Call method for Ylms.

        This returns ylms based on requested :math:`(l,m)` values and viewing
        angles.

        args:
            l_in: :math:`l` values requested.
            m_in: :math:`m` values requested.
            theta: Polar viewing angle.
            phi: Azimuthal viewing angle.

        Returns:
            Complex array of Ylm values. If theta and phi are arrays, the returned array will have shape `theta.shape + l_in.shape` (or `theta.shape + 2*l_in.shape` if `include_minus_m=True`).
        """
        if include_minus_m is None:
            include_minus_m = self.include_minus_m

        if bounds_check:
            assert self.xp.all(abs(self.spin_weight) <= l_in) and self.xp.all(self.xp.abs(m_in) <= l_in), f"Invalid s, l, m values: s={self.spin_weight}, l={l_in}, m={m_in}"

        if isinstance(l_in, int) or isinstance(m_in, int):
            assert isinstance(l_in, int) and isinstance(m_in, int)
            ylm_pos = self.wigner.sYlms_from_spherical_coordinates(self.spin_weight, np.array([l_in]), np.array([m_in]), theta, phi)[0]
            if include_minus_m:
                return ylm_pos, self.wigner.sYlms_from_spherical_coordinates(self.spin_weight, np.array([l_in]), np.array([-m_in]), theta, phi)[0]
            else:
                return ylm_pos

        # if assuming positive m, repeat entries for negative m
        # this will duplicate m = 0
        if include_minus_m:
            l = self.xp.zeros(2 * l_in.shape[0], dtype=int)
            m = self.xp.zeros(2 * l_in.shape[0], dtype=int)

            l[: l_in.shape[0]] = l_in
            l[l_in.shape[0] :] = l_in

            m[: l_in.shape[0]] = m_in
            m[l_in.shape[0] :] = -m_in

        # if not, just l_in, m_in
        else:
            l = l_in
            m = m_in

        # the function only works with CPU allocated arrays
        # if l and m are cupy arrays, turn into numpy arrays
        try:
            l = l.get()
            m = m.get()

        except AttributeError:
            pass

        # out = np.zeros(len(l), dtype=np.complex128)
        # get ylm arrays and cast back to cupy if using cupy/GPUs
        return self.xp.asarray(
            self.wigner.sYlms_from_spherical_coordinates(self.spin_weight, l, m, theta, phi)
        )
