#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: Copyright (c) 2025 Jean ACKER.
# SPDX-License-Identifier: Apache-2.0

"""
Classes and functions relating to reconfigurable intelligent surfaces
"""

from abc import ABC
from abc import abstractmethod
from typing import Optional
import tensorflow as tf
import matplotlib.pyplot as plt

from rt.radio_devices import RadioDevice
from rt.scene_object import SceneObject
from rt.utils.irs_utils import _top3_abs, _lagrange3, \
    expand_to_rank_dr, rotate_dr, normalize_dr, outer_dr


from typing_extensions import Tuple, Self                   
import mitsuba as mi
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR
from scipy.constants import speed_of_light
import drjit as dr
import numpy as np

class CellGrid():
    # pylint: disable=line-too-long
    r"""
    Class defining a cell grid that determines the physical structure of a RIS

    The cell grid specifies the location of unit cells within the y-z plane
    assuming a homogenous spacing of 0.5. The actual positions are computed by
    multiplying the cell positions by the wavelength and rotating them
    according to the RIS' orientation.

    A cell grid must have at least three columns and rows to ensure
    that discrete phase and amplitude profiles of the RIS can be interpolated.

    Parameters
    ----------
    num_rows : int
        Number of rows. Must at least be equal to three.

    num_cols : int
        Number of columns. Must at least be equal to three.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    """
    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 frequency: float):

        if num_rows < 3 or num_cols < 3:
            raise ValueError("num_rows and num_cols must be >= 3")
        self._num_rows = int(num_rows)
        self._num_cols = int(num_cols)

        self._cell_y_positions = dr.arange(dr.cuda.Float32, self.num_cols) 
        self._cell_y_positions -= dr.cuda.Float((self.num_cols-1.)/2.)

        self._cell_z_positions = dr.arange(dr.cuda.Float32, self._num_rows-1, -1, -1)
        self._cell_z_positions -= dr.cuda.Float((self.num_rows-1.)/2.)
        
        ny = dr.shape(self.cell_y_positions)[0]
        nz = dr.shape(self.cell_z_positions)[0]

        """
        z_grid = dr.tile(self.cell_y_positions, nz)  # shape: (nz*ny,)
        y_grid = dr.repeat(self.cell_z_positions, ny)    # shape: (nz*ny,)

        self._cell_positions = dr.empty(dr.cuda.TensorXf, (2,ny*nz))
        self._cell_positions[0,:] = z_grid
        self._cell_positions[1,:] = y_grid
        """
        z_grid,y_grid = dr.meshgrid(self._cell_y_positions, self._cell_z_positions)
        self._cell_positions = dr.empty(dr.cuda.TensorXf, (2,ny*nz))
        self._cell_positions[0,:] = z_grid
        self._cell_positions[1,:] = y_grid
        
        
        self._frequency = frequency
 

    @property
    def num_rows(self):
        r"""
        int : Number of rows
        """
        return self._num_rows

    @property
    def num_cols(self):
        r"""
        int : Number of columns
        """
        return self._num_cols

    @property
    def num_cells(self):
        r"""
        int : Number of cells
        """
        return self.num_rows * self.num_cols

    @property
    def cell_positions(self):
        r"""
        [num_cells, 2], tf.float : Cell positions ordered from
            top-to-bottom left-to-right
        """
        return self._cell_positions

    @property
    def cell_y_positions(self):
        r"""
        [num_cols], tf.float : y-coordinates of cells ordered
            from left-to-right
        """
        return self._cell_y_positions

    @property
    def cell_z_positions(self):
        r"""
        [num_rows], tf.float : z-coordinates of cells ordered
            from top-to-bottom
        """
        return self._cell_z_positions

    @property
    def frequency(self):
        """
        :py:class:`mi.Float` : Get/set the carrier frequency [Hz]
        """
        return self._frequency

    @frequency.setter
    def frequency(self, f):
        if f <= 0.0:
            raise ValueError("Frequency must be positive")
        self._frequency = mi.Float(f)


    @property
    def wavelength(self):
        """
         :py:class:`mi.Float` :  Wavelength [m]
        """
        return speed_of_light / self.frequency

    @property
    def wavenumber(self):
        """
         :py:class:`mi.Float` :  Wavelength [m]
        """
        return 2*np.pi/self.wavelength

class Profile(ABC):
    # pylint: disable=line-too-long
    r"""Abstract class defining a phase/amplitude profile of a RIS

    A Profile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def num_modes(self):
        r"""
        int : Number of reradiation modes
        """
        pass

    @abstractmethod
    def __call__(self, points, mode=None, return_grads=False):
        r"""
        Returns the profile values, gradients and Hessians at given points

        Input
        -----
        points : tf.float, [num_samples, 2]
            Tensor of 2D coordinates defining the points on the RIS at which
            the profile should be evaluated.
            Defaults to `None`. In this case, the values for all unit cells
            are returned.

        mode : int | `None`
            Reradiation mode to be considered.
            Defaults to `None`. In this case, the values for all modes
            are returned.

        return_grads : bool
            If `True`, also the first- and second-order derivatives are
            returned.
            Defaults to `False`.

        Output
        ------
        values : [num_modes, num_samples] or [num_samples], tf.float
            Interpolated profile values at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
            Gradients of the interpolated profile values
            at the sample positions. Only returned if `return_grads` is `True`.

        hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
            Hessians of the interpolated profile values
            at the sample positions. Only returned if `return_grads` is `True`.
        """
        pass

class AmplitudeProfile(Profile):
    # pylint: disable=line-too-long
    r"""Abstract class defining an amplitude profile of a RIS

    An AmplitudeProfile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    @property
    @abstractmethod
    def mode_powers(self):
        r"""
        [num_modes], tf.float: Relative power of reradiation modes
        """
        pass

class PhaseProfile(Profile):
    # pylint: disable=line-too-long
    r"""Abstract class defining a phase profile of a RIS

    A PhaseProfile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    pass

class DiscreteProfile(Profile):
    # pylint: disable=line-too-long
    r"""Class defining a discrete phase/amplitude profile of a RIS

    A DiscreteProfile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    cell_grid : :class:`~sionna.rt.CellGrid`
        Defines the physical structure of the RIS

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    values : tf.float or tf.Variable, [num_modes, num_rows, num_cols]
        Values of the discrete profile for each reradiation mode
        and unit cell. `num_rows` and `num_cols` are defined by the
        `cell_grid`.
        Defaults to `None`.

    interpolator : :class:`~sionna.rt.ProfileInterpolator`
        Instance of a `ProfileInterpolator` that interpolates the
        discrete values of the profile to a continuous profile
        which is defined at any point on the RIS.
        Defaults to `None`. In this case, the
        :class:`~sionna.rt.LagrangeProfileInterpolator` will be used.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    def __init__(self,
                 cell_grid,
                 num_modes=1,
                 values=None,
                 interpolator=None):

        super().__init__()
        self._cell_grid = cell_grid
        self._num_modes = dr.cuda.ad.Int(num_modes)
        self._num_modes_int = int(num_modes)
        if values is None:
            self._values = None
        else:
            self.values = values
        if interpolator is None:
            self._interpolator = LagrangeProfileInterpolator(self)
        else:
            self._interpolator = interpolator

    @property
    def shape(self):
        r"""
        tf.TensorShape : Shape of the tensor holding the values of
            the discrete profile
        """
        return (int(self.cell_grid.num_rows),int(self.cell_grid.num_cols), int(self.num_modes[0]))
    

    @property
    def values(self):
        r"""
        [shape], tf.float : Set/get the discrete values of the profile for each
            reradiation mode
        """
        
        return self._values

    @values.setter
    def values(self, v):
        # Shape check
        if (v.shape != self.shape):
            raise ValueError(f"`values` must have shape {self.shape}")

        # Dtype check
        expected_type = dr.cuda.ad.TensorXf
        if dr.type_v(v) != expected_type:
            try:
                v = expected_type(v) 
            except Exception:
                raise TypeError(f"`values` must have dtype={expected_type}")
        self._values = v

    @property
    def num_modes(self):
        r"""
        int : Number of reradiation modes
        """
        return self._num_modes

    @property
    def cell_grid(self):
        r"""
        :class:`~sionna.rt.CellGrid` : Defines the physical
            structure of the RIS
        """
        return self._cell_grid

    @property
    def spacing(self):
        r"""
        tf.float: Element spacing [m] corresponding to
            half a wavelength
        """
        return self.cell_grid.wavelength/dr.cuda.Float(2)

    def show(self, i=0, path="results", mode=0):
        r"""Visualizes the profile as a 3D plot

        Input
        ------
        mode : int | `None`
            Reradation mode to be shown.
            Defaults to 0.

        Output
        ------
        : :class:`matplotlib.pyplot.Figure`
            3D plot of the profile
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        y,z = np.meshgrid(self.cell_grid.cell_y_positions*self.spacing,
                          self.cell_grid.cell_z_positions*self.spacing)


        ax.plot_surface(y, z, self.values[:,:,mode], cmap='viridis')
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        if isinstance(self, PhaseProfile):
            plt.title(r"Phase profile $\chi(y, z)$")
            plt.savefig(f"{path}/phase_profile_{i}.png")
        if isinstance(self, AmplitudeProfile):
            plt.title(r"Amplitude profile $A(y, z)$")
            plt.savefig(f"{path}/amplitude_profile_{i}.png")
        return fig

    def __call__(self, points=None, mode=None, return_grads=False):
        if points is None:
            if mode is not None:
                values = self.values[...,mode]
            else:
                values = self.values
            return values
        
        else:
            return self._interpolator(points, mode, return_grads)

class ProfileInterpolator(ABC):
    r"""
    Abstract class defining an interpolator of a discrete profile

    A ProfileInterpolator instance is a callable that interpolate
    the discrete profile to specified points. Optionally, the
    gradients and Hessians are returned.

    Parameters
    ----------
    discrete_profile : :class:`~sionna.rt.DiscreteProfile`
        Discrete profile to be interpolated

    Input
    -----
    points : [num_samples, 2], tf.float
        Positions at which to interpolate the profile

    mode : int | `None`
        Mode of the profile to interpolate. If `None`.
        all modes are interpolated.
        Defaults to `None`.

    return_grads : bool
        If `True`, gradients and Hessians are computed.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions

    hessians : [num_modes, num_samples, 3, 3] or [num_samples,3,3], tf.float
        Hessians of the interpolated profile values
        at the sample positions
    """
    def __init__(self, discrete_profile):
        self._discrete_profile = discrete_profile

    @property
    def spacing(self):
        r"""
        tf.float: Element spacing [m] corresponding to
            half a wavelength
        """
        return self._discrete_profile.cell_grid.wavelength/dr.cuda.Float(2)

    @property
    def cell_y_positions(self):
        r"""
        [num_cols], tf.float : y-coordinates of cells ordered
            from left-to-right
        """
        return self._discrete_profile.cell_grid.cell_y_positions*self.spacing

    @property
    def cell_z_positions(self):
        r"""
        [num_rows], tf.float : z-coordinates of cells ordered
            from top-to-bottom
        """
        return self._discrete_profile.cell_grid.cell_z_positions*self.spacing

    @property
    def num_rows(self):
        r"""
        int : Number of rows
        """
        return self._discrete_profile.cell_grid.num_rows

    @property
    def num_cols(self):
        r"""
        int : Number of columns
        """
        return self._discrete_profile.cell_grid.num_cols

    @property
    def values(self):
        r"""
        [shape], tf.float : Discrete values of the profile for each
            reradiation mode and unit cell
        """
        return self._discrete_profile.values

    @abstractmethod
    def __call__(self, points, mode=None, return_grads=False):
        r"""
        Interpolates the discrete profile to specified points

        Optionally, the gradients and Hessians are returned.

        Input
        -----
        points : [num_samples, 2], tf.float
            Positions at which to interpolate the profile

        mode : int | `None`
            Mode of the profile to interpolate. If `None`.
            all modes are interpolated.
            Defaults to `None`.

        return_grads : bool
            If `True`, gradients and Hessians are computed.
            Defaults to `False`.

        Output
        ------
        values : [num_modes, num_samples] or [num_samples], tf.float
            Interpolated profile values at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
            Gradients of the interpolated profile values
            at the sample positions

        hessians : [num_modes, num_samples, 3, 3] or [num_samples,3,3], tf.float
            Hessians of the interpolated profile values
            at the sample positions
        """
        pass

class LagrangeProfileInterpolator(ProfileInterpolator):
    # pylint: disable=line-too-long
    r"""
    Class defining a :class:`~sionna.rt.ProfileInterpolator` using Lagrange polynomials

    The class instance is a callable that interpolates a discrete profile
    at arbitrary positions using two-dimensional 2nd-order Lagrange interpolation.

    A discrete profile :math:`P(y_i,z_j)\in\mathbb{R}` defined on
    a grid of points :math:`y_i,z_j` for :math:`i,j \in [1,2,3]` is
    interpolated to position :math:`y,z` as

    .. math::
        \begin{align}
            P(y,z) &= \sum_{i,j} P(y_i,z_j) \ell_{i,y}(y) \ell_{j,z}(z)
        \end{align}

    where :math:`\ell_{i,y}(y)`, :math:`\ell_{j,z}(z)` are the
    one-dimensional 2nd-order Lagrange polynomials, defined
    as

    .. math::
        \begin{align}
            \ell_{i,y}(y) &= \prod_{j \ne i} \frac{y-y_j}{y_i-y_j} \\
            \ell_{j,z}(z) &= \prod_{i \ne j} \frac{z-z_i}{z_j-z_i}.
        \end{align}

    Note that the formulation above assumes for simplicity only a 3x3 grid
    of points. However, the implementation finds for every
    position the closest 3x3 grid points of the discrete profile
    which are used for interpolation.

    In order to compute spatial gradients and Hessians, we extend the the profile
    with a dummy :math:`x` dimension, i.e., :math:`P(x,y,z)=P(y,z)`, such that

    .. math::
        \begin{align}
            \nabla P(x,y,z) &= \begin{bmatrix} 0, \frac{\partial P(x,y,z)}{\partial y}, \frac{\partial P(x,y,z)}{\partial z}  \end{bmatrix}^{\textsf{T}}\\
            H_P(x,y,z) &= \begin{bmatrix} 0 & 0                                                 & 0 \\
                                            0 & \frac{\partial^2 P(x,y,z)}{\partial y^2}          & \frac{\partial^2 P(x,y,z)}{\partial y \partial z} \\
                                            0 & \frac{\partial^2 P(x,y,z)}{\partial z \partial y} & \frac{\partial^2 P(x,y,z)}{\partial z^2}
                            \end{bmatrix}
        \end{align}.

    Parameters
    ----------
    discrete_profile : :class:`~sionna.rt.DiscreteProfile`
        Discrete profile to be interpolated

    Input
    -----
    points : [num_samples, 2], tf.float
        Positions at which to interpolate the profile

    mode : int | `None`
        Mode of the profile to interpolate. If `None`,
        all modes are interpolated.
        Defaults to `None`.

    return_grads : bool
        If `True`, gradients and Hessians are computed.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions
    """
    @staticmethod
    def lagrange_polynomials(x, x_i, return_derivatives=True):
        """
        x:    [batch_size] drjit.Float (or dr.cuda.Float)
        x_i:  [3, batch_size] drjit.Float
        """
        sample_diff = x - x_i 
        sample_prod_0 = sample_diff[1] * sample_diff[2] #(4,)
        sample_prod_1 = sample_diff[0] * sample_diff[2] #(4,)
        sample_prod_2 = sample_diff[0] * sample_diff[1] #(4,)
        sample_prods = dr.cuda.Array3f([sample_prod_0, sample_prod_1, sample_prod_2])
        sample_prods = dr.cuda.TensorXf(sample_prods)
        support_diffs = x_i[dr.newaxis, ...] - x_i[:, dr.newaxis, :] #[3,3,B]
        support_diffs = dr.select(support_diffs == 0, 1.0, support_diffs)
        support_prods = dr.prod(support_diffs, axis=0)

        # Lagrange polynomials
        lagrange = sample_prods / support_prods
        
        if not return_derivatives:
            return lagrange
        else:
            # Compute sums of differences
            sample_sum_0 = sample_diff[1] + sample_diff[2]
            sample_sum_1 = sample_diff[0] + sample_diff[2]
            sample_sum_2 = sample_diff[0] + sample_diff[1]  
            #sample_sums = dr.empty()
            sample_sums  = dr.cuda.Array3f([sample_sum_0, sample_sum_1, sample_sum_2])
            sample_sums = dr.cuda.TensorXf(sample_sums)
            
            # First-order derivatives
            deriv_1st = sample_sums / support_prods
            # Second-order derivatives
            deriv_2nd = dr.full(dr.cuda.TensorXf, 2.0, dr.shape(support_prods)) / support_prods
            return lagrange, deriv_1st, deriv_2nd
    
    @staticmethod
    def lagrange_polynomials_old(x,
                             x_i,
                             return_derivatives=True):
        # pylint: disable=line-too-long
        r"""
        Compute the 2nd-order Lagrange polynomials

        Optionally, the first- and second-order derivatives are returned.

        The 2nd-order Lagrange polynomials :math:`\ell_j(x)`, :math:`j=1,2,3`,
        for position :math:`x\in\mathbb{R}` are computed using three distinct
        support positions :math:`x_i` for :math:`i=1,2,3`:

        .. math::
            \begin{align}
                \ell_j(x) &= \prod_{\substack{1\leq i \leq 3 \\ i \ne j}} \frac{x-x_i}{x_j-x_i}.
            \end{align}

        Their first- and second-order derivatives are then respectively given as

        .. math::
            \begin{align}
                \ell'_j(x)  &= \left(\sum_{i \ne j} x-x_i\right) \left(\prod_{i \ne j} x_j-x_i\right)^{-1} \\
                \ell''_j(x) &= 2 \left(\prod_{i \ne j} x_j-x_i\right)^{-1}.
            \end{align}

        Input
        -----
        x : [batch_size], tf.float
            Sample positions

        x_i : [batch_size, 3], tf.float
            Support positions for every sample position

        return_derivatives : bool
            If `True`, also the first- and second-order derivatives
            of the Lagrange polynomials are returned.
            Defaults to `True`.

        Output
        ------
        l_i : [batch_size, 3], tf.float
            Lagrange polynomials for each sample position

        deriv_1st : [batch_size, 3], tf.float
            First-order derivatives for each sample position.
            Only returned if `return_derivatives` is `True`.

        deriv_2nd : [batch_size, 3], tf.float
            Second-order derivatives for each sample position.
            Only returned if `return_derivatives` is `True`.
        """

        # Compute products of differences of the sample and support points
        sample_diff = tf.expand_dims(x, 1) - x_i
        sample_prod_0 = sample_diff[:,1]*sample_diff[:,2]
        sample_prod_1 = sample_diff[:,0]*sample_diff[:,2]
        sample_prod_2 = sample_diff[:,0]*sample_diff[:,1]
        sample_prods = tf.stack([sample_prod_0, sample_prod_1, sample_prod_2], -1)

        # Compute products of differences of support points
        support_diffs = tf.expand_dims(x_i, -1) - tf.expand_dims(x_i, -2)
        support_diffs = tf.where(support_diffs==0, 1., support_diffs)
        support_prods = tf.reduce_prod(support_diffs, axis=-1)

        # Compute Lagrange polynomials
        lagrange = sample_prods/support_prods

        if not return_derivatives:
            return lagrange
        else:
            # Compute sums of differences
            sample_sum_0 = sample_diff[:,1] + sample_diff[:,2]
            sample_sum_1 = sample_diff[:,0] + sample_diff[:,2]
            sample_sum_2 = sample_diff[:,0] + sample_diff[:,1]
            sample_sums = tf.stack([sample_sum_0, sample_sum_1, sample_sum_2],
                                    -1)
            # Compute first-order derivatives
            deriv_1st = sample_sums/support_prods

            # Compute second-order derivatives
            deriv_2nd = tf.cast(2, support_prods.dtype)/support_prods

            return lagrange, deriv_1st, deriv_2nd
    
    def __call__(self,
                    points: mi.Vector2f,
                    mode: Optional[int] = None,
                    return_grads: bool = False):
        """
        M -> num_modes
        B -> num_samples
        Inputs:
        - points : [B, 2], mi.Vector2f
            Points at which to interpolate the profile, in the local coordinate system of the IRS
        - mode : int | None
            If not None, only interpolate the specified mode
        - return_grads : bool
            If True, also return gradients and Hessians
        Returns:
        -values   : TensorXf (M, B)
        - grads    : TensorXf (M, B, 3)          (only if return_grads)
        - hessians : TensorXf (M, B, 3, 3)       (only if return_grads)
        """
        # ---- sizes ----
        B  = dr.width(points.x)          # num_samples
        Ny = int(self.num_rows)
        Nz = int(self.num_cols)
        M  = int(self.values.shape[-1])   # num_modes

        # 1) 3 nearest supports along y & z
        yi0, yi1, yi2, yv0, yv1, yv2 = _top3_abs(points.x, self.cell_y_positions)
        zi0, zi1, zi2, zv0, zv1, zv2 = _top3_abs(points.y, self.cell_z_positions)

        # 2) 1D Lagrange polynomials (and derivs)
        if not return_grads:
            Ly = _lagrange3(points.x, yv0, yv1, yv2, return_grads=False)  # tuple of 3 mi.Float[B]
            Lz = _lagrange3(points.y, zv0, zv1, zv2, return_grads=False)
        else:
            (Ly, d1y, d2y) = _lagrange3(points.x, yv0, yv1, yv2, return_grads=True)
            (Lz, d1z, d2z) = _lagrange3(points.y, zv0, zv1, zv2, return_grads=True)
             
        # 3) 3×3 flat indices per lane: flat = z*Ny + y (y fastest) 
        # as the discrete values of the profile are flatten y fastest
        yi = [yi0, yi1, yi2]
        zi = [zi0, zi1, zi2]
        flat = [[None]*3 for _ in range(3)]
        for I in range(3):
            yI = mi.UInt32(yi[I])
            for J in range(3):
                zJ = mi.UInt32(zi[J])
                flat[I][J] = yI * mi.UInt32(Nz) + zJ            # mi.UInt32[B]

        # 4) weight products (per lane) reused across all modes
        w   = [[Ly[J] * Lz[I] for J in range(3)] for I in range(3)]             # value weights
        if return_grads:
            wy  = [[d1y[J] * Lz[I] for J in range(3)] for I in range(3)]        # d/dy weights
            wz  = [[Ly[J] * d1z[I] for J in range(3)] for I in range(3)]        # d/dz weights
            wyy = [[d2y[J] * Lz[I] for J in range(3)] for I in range(3)]        # d2/dy2
            wzz = [[Ly[J] * d2z[I] for J in range(3)] for I in range(3)]        # d2/dz2
            wyz = [[d1y[J] * d1z[I] for J in range(3)] for I in range(3)]       # d2/dy dz

        # 5) flat value buffer (mi.Float length = M*Ny*Nz)
        if isinstance(self.values, dr.cuda.ad.TensorXf):
            vals_flat = dr.reshape(dr.cuda.ad.TensorXf, self.values, [-1], order='A').array
        else:
            vals_flat = self.values 

        stride = 1

        # 6) allocate outputs 
        values = dr.zeros(dr.cuda.ad.TensorXf, (M, B))
        if return_grads:
            grads    = dr.zeros(dr.cuda.ad.TensorXf, (M, B, 3))
            hessians = dr.zeros(dr.cuda.ad.TensorXf, (M, B, 3, 3))

        # 7) per-mode accumulation with precomputed supports/weights
        for m in range(M):
            base = mi.UInt32(m * stride)

            # gather 9 supports once for this mode
            g = [[None]*3 for _ in range(3)]
            for I in range(3):
                for J in range(3):
                    g[I][J] = dr.gather(mi.Float, vals_flat, base + flat[I][J]*M)  # mi.Float[B]

            val = mi.Float(0.0)
            for I in range(3):
                for J in range(3):
                    val += w[I][J] * g[I][J]

            # write row m
            values[m, :] = val

            if return_grads:
                dvy = mi.Float(0.0)
                dvz = mi.Float(0.0)
                dyy = mi.Float(0.0)
                dzz = mi.Float(0.0)
                dyz = mi.Float(0.0)
                for I in range(3):
                    for J in range(3):
                        gij = g[I][J]
                        dvy += wy[I][J]  * gij
                        dvz += wz[I][J]  * gij
                        dyy += wyy[I][J] * gij
                        dzz += wzz[I][J] * gij
                        dyz += wyz[I][J] * gij

                grads[m, :, 0] = 0.0
                grads[m, :, 1] = dvy
                grads[m, :, 2] = dvz

                # Hessian 3×3 with yz-block filled
                hessians[m, :, 0, 0] = 0.0; hessians[m, :, 0, 1] = 0.0; hessians[m, :, 0, 2] = 0.0
                hessians[m, :, 1, 0] = 0.0; hessians[m, :, 1, 1] = dyy; hessians[m, :, 1, 2] = dyz
                hessians[m, :, 2, 0] = 0.0; hessians[m, :, 2, 1] = dyz; hessians[m, :, 2, 2] = dzz

        if return_grads:
            return values, grads, hessians
        else:
            return values
    
class DiscreteAmplitudeProfile(DiscreteProfile, AmplitudeProfile):
    # pylint: disable=line-too-long
    r"""Class defining a discrete amplitude profile of a RIS

    A discrete amplitude profile :math:`A_m` assigns to
    each of its units cells a possibly different amplitude value.
    Multiple reradiation modes can be obtained by super-positioning
    of profiles. The relative power of reradiation modes can
    be controlled via the reradiation coefficients :math:`p_m`.

    See :ref:`ris_primer` for more details.

    A class instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    cell_grid : :class:`~sionna.rt.CellGrid`
        Defines the physical structure of the RIS

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    values : tf.float or tf.Variable, [num_modes, num_rows, num_cols]
        Amplitude values for each reradiation mode
        and unit cell. `num_rows` and `num_cols` are defined by the
        `cell_grid`.
        Defaults to `None`.

    mode_powers : tf.float, [num_modes]
        Relative powers or reradition coefficients of reradiation modes.
        Defaults to `None`. In this case, all reradiation modes get
        an equal fraction of the total power.

    interpolator : :class:`~sionna.rt.ProfileInterpolator`
        Determines how the discrete values of the profile
        are interpolated to a continuous profile
        which is defined at any point on the RIS.
        Defaults to `None`. In this case, the
        :class:`~sionna.rt.LagrangeProfileInterpolator` will be used.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    def __init__(self,
                 cell_grid,
                 num_modes=1,
                 values=None,
                 mode_powers=None,
                 interpolator=None,
                 dtype=tf.complex64):
        super().__init__(cell_grid=cell_grid,
                         num_modes=num_modes,
                         values=values,
                         interpolator=interpolator)

        if values is None:
            values = dr.ones(dr.cuda.ad.TensorXf, self.shape)

        if mode_powers is None:
            mode_powers = 1/dr.cuda.ad.Float(self.num_modes) * \
                          dr.ones(dr.cuda.ad.TensorXf, [self.num_modes[0]])
        self.mode_powers = mode_powers

    @property
    def mode_powers(self):
        return self._mode_powers

    @mode_powers.setter
    def mode_powers(self, v):
        vx = v if isinstance(v, dr.cuda.ad.TensorXf) else dr.cuda.ad.TensorXf(v)
        vx = dr.reshape(dr.cuda.ad.TensorXf, vx, [-1])
        if vx.shape[0] != self.num_modes:
            raise ValueError(f"`mode_powers` must have shape ({self.num_modes},), got {vx.shape}")
        self._mode_powers = vx

class DiscretePhaseProfile(DiscreteProfile, PhaseProfile):
    # pylint: disable=line-too-long
    r"""Class defining a discrete phase profile of a RIS

    A discrete phase profile :math:`\chi_m` assigns to
    each of its units cells a possibly different phase value.
    Multiple reradiation modes can be created by super-positioning
    of phase profiles.

    See :ref:`ris_primer` in the Primer on Electromagnetics for more details.

    A class instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    cell_grid : :class:`~sionna.rt.CellGrid`
        Defines the physical structure of the RIS

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    values : tf.float or tf.Variable, [num_modes, num_rows, num_cols]
        Phase values [rad] for each reradiation mode
        and unit cell. `num_rows` and `num_cols` are defined by the
        `cell_grid`.
        Defaults to `None`.

    interpolator : :class:`~sionna.rt.ProfileInterpolator`
        Determines how the discrete values of the profile
        are interpolated to a continuous profile
        which is defined at any point on the RIS.
        Defaults to `None`. In this case, the
        :class:`~sionna.rt.LagrangeProfileInterpolator` will be used.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], tf.float
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    def __init__(self,
                 cell_grid,
                 num_modes=1,
                 values=None,
                 interpolator=None,):
        super().__init__(cell_grid=cell_grid,
                         num_modes=num_modes,
                         values=values,
                         interpolator=interpolator)

        if values is None:
            self.values = dr.zeros(dr.cuda.ad.TensorXf, self.shape)

class RIS(RadioDevice, SceneObject):
    # pylint: disable=line-too-long
    r"""
    Class defining a reconfigurable intelligent surface (RIS)

    A RIS consists of a planar arrangement of unit cells
    with :math:`\lambda/2` spacing.
    It's :class:`~sionna.rt.PhaseProfile` :math:`\chi_m` and
    :class:`~sionna.rt.AmplitudeProfile` :math:`A_m` can be
    configured after the RIS is instantiated. Both together
    define the spatial modulation coefficient :math:`\Gamma` which
    determines how the RIS reflects electro-magnetic waves.

    See :ref:`ris_primer` in the Primer on Electromagnetics for
    more details or have a look at the `tutorial notebook <https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_RIS.html>`_.

    An RIS instance is a callable that computes the spatial modulation coefficient
    and gradients/Hessians of the underlying phase profile for provided
    points on the RIS' surface.

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    num_rows : int
        Number of rows. Must at least be equal to three.

    num_cols : int
        Number of columns. Must at least be equal to three.

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0]. In this case, the normal vector of
        the RIS points towards the positive x-axis.

    velocity : [3], float
        Velocity vector [m/s]. Used for the computation of
        path-specific Doppler shifts.

    look_at : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RIS` | :class:`~sionna.rt.Camera` | `None`
        A position or the instance of a :class:`~sionna.rt.Transmitter`,
        :class:`~sionna.rt.Receiver`, :class:`~sionna.rt.RIS`, or :class:`~sionna.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    color : [3], float
        Defines the RGB (red, green, blue) ``color`` parameter for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.
        Defaults to `[0.862,0.078,0.235]`.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.

    Input
    -----
    points : tf.float, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the spatial modulation profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    gamma : [num_modes, num_samples] or [num_samples], tf.complex
        Spatial modulation coefficient at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], tf.float
        Gradients of the interpolated phase profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , tf.float
        Hessians of the interpolated phase profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    def __init__(self,
                name: str,
                position: mi.Point3f,
                num_rows: int,
                num_cols: int,
                frequency: float,
                num_modes: int = 1,
                orientation: mi.Point3f | None = None,
                velocity: mi.Vector3f | None = None,
                look_at: mi.Point3f | Self | None = None,
                color: Tuple[float, float, float] = DEFAULT_TRANSMITTER_COLOR,
                display_radius: float | None = None,
                dtype=tf.complex64):

        self._rdtype = tf.float32
        # Initialize the parent classes
        # RadioDevice and SceneObject inherit from Object
        # Python will initialize in the following order:
        # RadioDevice->SceneObject->Object
        super().__init__(name=name,position=position,orientation=orientation,look_at=look_at,color=color, display_radius=display_radius)
        position2 = [self.position[0], self.position[1], self.position[2]]
        self.position2 = dr.cuda.ad.TensorXf(position2)
        orientation = [self.orientation[0], self.orientation[1], self.orientation[2]]
        self.orientation2 = dr.cuda.ad.TensorXf(orientation)
        self.frequency = frequency
        # Set velocity vector
        #self.velocity = tf.cast(velocity, dtype=dtype.real_dtype)

        if num_rows < 3 or num_cols < 3:
            raise ValueError("num_rows and num_cols must be >= 3")

        # Set immutable properties
        self._num_modes = int(num_modes)
        self._cell_grid = CellGrid(num_rows, num_cols, frequency)#, self._dtype)

        # Init amplitude profile
        self.amplitude_profile = DiscreteAmplitudeProfile(self.cell_grid,
                                                     num_modes=self.num_modes,)
                                                     #dtype=self._dtype)

        # Init phase profile
        self.phase_profile = DiscretePhaseProfile(self.cell_grid,
                                             num_modes=self.num_modes)

    @property
    def cell_grid(self):
        r"""
        :class:`~sionna.rt.CellGrid` : Defines the physical
            structure of the RIS
        """
        return self._cell_grid

    @property
    def cell_positions(self):
        r"""
        [num_cells, 2], tf.float : Cell positions in the
            local coordinate system (LCS) of the RIS, ordered
            from top-to-bottom left-to-right.
        DR : [2, num_cells]
        """
        return self.cell_grid.cell_positions*self.spacing

    @property
    def cell_world_positions(self):
        r"""
        [num_cells, 3], tf.float : Cell positions in the
            global coordinate system (GCS) of the RIS, ordered
            from top-to-bottom left-to-right.
        """
        x_coord = dr.zeros(dr.cuda.ad.Array1f, [1, self.num_cells])
        #[1,num_cells] & [2,num_cells] --> [3,num_cells]
        temp1 = dr.reshape(dr.cuda.ad.Array1f, self.cell_positions[0], (1, -1))
        temp2 = dr.reshape(dr.cuda.ad.Array1f, self.cell_positions[1], (1, -1)) 
        pos = dr.empty(dr.cuda.ad.TensorXf, (3, self.num_cells))
        pos[0,:] = x_coord
        pos[1,:] = temp1
        pos[2,:] = temp2
        pos = rotate_dr(pos, self.orientation2)
        pos += self.position2
        return pos

    @property
    def world_normal(self) -> mi.Vector3f:
        """mi.Vector3f: RIS normal in world coordinates."""
        # Local-frame normal of the RIS plane (pointing +x)
        n_hat = mi.Vector3f(1.0, 0.0, 0.0)

        # Ensure we have a Vector3f of Euler angles (radians)
        ori = mi.Vector3f(self.orientation)


        # Euler XYZ (roll=x, pitch=y, yaw=z). Change order if your convention differs.
        cx, sx = dr.cos(ori.x), dr.sin(ori.x)
        cy, sy = dr.cos(ori.y), dr.sin(ori.y)
        cz, sz = dr.cos(ori.z), dr.sin(ori.z)

        Rx = mi.Matrix3f([[1.0, 0.0, 0.0],
                          [0.0,  cx, -sx],
                          [0.0,  sx,  cx]])

        Ry = mi.Matrix3f([[ cy, 0.0, sy],
                          [0.0, 1.0, 0.0],
                          [-sy, 0.0, cy]])

        Rz = mi.Matrix3f([[ cz, -sz, 0.0],
                          [ sz,  cz, 0.0],
                          [0.0, 0.0, 1.0]])

        # Apply rotations in XYZ order: v_world = Rz * Ry * Rx * v_local
        R = Rz @ Ry @ Rx
        n_world = R @ n_hat
        return dr.normalize(n_world)

    @property
    def num_rows(self):
        r"""
        int : Number of rows
        """
        return self.cell_grid.num_rows

    @property
    def num_cols(self):
        r"""
        int : Number of columns
        """
        return self.cell_grid.num_cols

    @property
    def num_cells(self):
        r"""
        int : Number of cells
        """
        return self.num_rows*self.num_cols

    @property
    def num_modes(self):
        r"""
        int : Number of reradiation modes
        """
        return self._num_modes

    @property
    def spacing(self):
        r"""
        tf.float: Element spacing [m] corresponding to
            half a wavelength
        """
        return self.cell_grid.wavelength/dr.cuda.Float(2)

    @property
    def size(self) -> mi.Point2f:
        """Size of the RIS"""
        # replicate the ints across the same width as spacing
        W = dr.width(self.spacing)  # lane count of the JIT float
        ncols = dr.full(mi.Float, float(self.num_cols), W)
        nrows = dr.full(mi.Float, float(self.num_rows), W)

        width  = self.spacing * ncols
        height = self.spacing * nrows
        return mi.Point2f(width, height)

    @property
    def amplitude_profile(self):
        r"""
        :class:`~sionna.rt.AmplitudeProfile` : Set/get amplitude profile
        """
        return self._amplitude_profile

    @amplitude_profile.setter
    def amplitude_profile(self, v):
        if not isinstance(v, AmplitudeProfile):
            raise ValueError("Not a valid AmplitudeProfile")
        self._amplitude_profile = v

    @property
    def phase_profile(self):
        r"""
        :class:`~sionna.rt.PhaseProfile` : Set/get phase profile
        """
        return self._phase_profile

    @phase_profile.setter
    def phase_profile(self, v):
        if not isinstance(v, PhaseProfile):
            raise ValueError("Not a valid PhaseProfile")
        self._phase_profile = v

    def uniform_init(self):
        self.phase_profile.values = dr.zeros(dr.cuda.ad.TensorXf, (self.num_rows, self.num_cols, self.num_modes))
        self.amplitude_profile.values = dr.ones(dr.cuda.ad.TensorXf, (self.num_rows, self.num_cols, self.num_modes))

        mode_powers = 1/dr.cuda.Float(self.num_modes) * \
                    dr.ones(dr.cuda.ad.TensorXf, (self.num_modes,))
        
        self.amplitude_profile.mode_powers = mode_powers

    def phase_gradient_reflector(self, sources, targets):
        # pylint: disable=line-too-long
        r"""
        Configures the RIS as ideal phase gradient reflector

        For an incoming direction :math:`\hat{\mathbf{k}}_i`
        and desired outgoing direction :math:`\hat{\mathbf{k}}_r`,
        the necessary phase gradient along the RIS with normal
        :math:`\hat{\mathbf{n}}` can be computed as
        (e.g., Eq.(12) [Vitucci24]_):

        .. math::
            \nabla\chi_m = k_0\left( \mathbf{I}- \hat{\mathbf{n}}\hat{\mathbf{n}}^\textsf{T} \right) \left(\hat{\mathbf{k}}_i - \hat{\mathbf{k}}_r  \right).

        The phase profile is obtained by assigning zero phase to the first
        unit cell and evolving the other phases linearly according to the gradient
        across the entire RIS.

        Multiple reradiation modes can be configured.

        The amplitude profile is set to one everywhere with a uniform relative
        power allocation across modes.

        Input
        -----
        sources : tf.float, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a source from which the incoming wave originates.

        targets : tf.float, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a target towards which the incoming wave should be
            reflected.
        """
        # Convert inputs to tensors
        sources = dr.cuda.ad.TensorXf(sources)
        targets = dr.cuda.ad.TensorXf(targets)
        sources = expand_to_rank_dr(sources, 2, -1)
        targets = expand_to_rank_dr(targets, 2, -1)
        shape = [3, self.num_modes]

        # Ensure the desired shape [num_modes, 3]
        for i, x in enumerate([sources, targets]):
            if not (x.shape[0] == shape[0]):# and x.shape[1] == shape[1]):
            #if not (tf.shape(x)==shape).numpy().all():
                msg = f"Wrong shape of input {i+1}. " + \
                      f"Expected {shape}, got {x.shape}"
                raise ValueError(msg)

        # Compute incoming and outgoing directions
        # [3, num_modes]
        k_i = dr.normalize(self.position2 - sources)
        k_r = dr.normalize(targets - self.position2)

        # Tangent projection operator - Eq.(10)
        # [3, 1]
        normal = dr.cuda.ad.TensorXf(self.world_normal)
        normal = dr.reshape(dr.cuda.ad.TensorXf, normal, (3,1))

        # [1, 3, 3]
        p = tf.eye(3, dtype=self._rdtype) 
        # [3, 3, 1]
        p = dr.cuda.ad.TensorXf(p.numpy())
        p = dr.reshape(dr.cuda.ad.TensorXf, p, (3,3,-1))
        p -= outer_dr(normal,normal)
        p_mat = dr.cuda.ad.Matrix3f(p)

        # Compute phase gradient - Eq.(12)
        # [3, num_modes] self.cell_grid.wavenumber
        
        grad = self.cell_grid.wavenumber * p_mat @ dr.cuda.ad.Array3f(k_i - k_r)    #dr.dot(k_i - k_r, p)


        # Rotate phase gradient to LCS of the RIS and keep y/z components
        # [num_modes, 1, 1, 2]
        grad = rotate_dr(grad, self.orientation2, inverse=True)[1:,:]
        grad = dr.reshape(dr.cuda.ad.TensorXf, grad, [2, 1, 1, self.num_modes])
        
        # Using the top-left cell as reference, compute the offsets
        # [1, num_rows, num_cols, 2] --> [2, num_rows, num_cols, 1]
        offsets = self.cell_positions - self.cell_positions[...,:1]
        offsets = dr.reshape(dr.cuda.ad.TensorXf, offsets, (2, self.num_rows, self.num_cols))

        offsets = offsets[...,dr.newaxis]
        # Compute phase profile based on the constant gradient assumption
        # [num_modes, num_rows, num_cols]
        phases = dr.sum(grad*offsets, axis=0)

        self.phase_profile.values = phases
        # Set a neutral amplitude profile
        #[num_cols, num_rows, num_samples]
        self.amplitude_profile.values = dr.ones(dr.cuda.ad.TensorXf, phases.shape)
        #[num_samples]
        mode_powers = 1/dr.cuda.Float(self.num_modes) * \
                    dr.ones(dr.cuda.ad.TensorXf, (self.num_modes,))
        

        self.amplitude_profile.mode_powers = mode_powers


    """
    def focusing_lens(self, sources, targets):
        # pylint: disable=line-too-long
        r
        Configures the RIS as focusing lens

        The phase profile is configured in such a way that
        the fields of all rays add up coherently at a specific
        point. In other words, the phase profile undoes the
        distance-based phase shift of every ray connecting a
        source to a target via a specific unit cell.

        For a source and target at positions
        :math:`\mathbf{s}` and :math:`\mathbf{t}`, the phase
        :math:`\chi_m(\mathbf{x})` of a unit cell located at :math:`\mathbf{x}`
        is computed as (e.g., Sec. IV-2 [Degli-Esposti22]_)

        .. math::
            \chi_m(\mathbf{x}) = k_0 \left(\lVert\mathbf{s}-\mathbf{x}\rVert + \lVert\mathbf{s}-\mathbf{t}\rVert\right).

        Multiple reradiation modes can be configured.

        The amplitude profile is set to one everywhere with a uniform relative
        power allocation across modes.

        Input
        -----
        sources : tf.float, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a source from which the incoming wave originates.

        targets : tf.float, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a target towards which the incoming wave should be
            reflected.
        
        # Convert inputs to tensors
        sources = dr.cuda.ad.TensorXf(sources)
        targets = dr.cuda.ad.TensorXf(targets)
        print("sources.shape: ", sources.shape)
        print("targets.shape: ", targets.shape)
        sources = expand_to_rank_dr(sources, 2, 0)
        targets = expand_to_rank_dr(targets, 2, 0)
        shape = [3, self.num_modes]

        # Ensure the desired shape [num_modes, 3]
        for i, x in enumerate([sources, targets]):
            if not (x.shape[0]==shape[0] and x.shape[1]==shape[1]):
                msg = f"Wrong shape of input {i+1}. " + \
                      f"Expected {shape}, got {x.shape}"
                raise ValueError(msg)

        # Compute incoming and outgoing distances
        # [num_modes, num_cells]
        print("self.cell_world_positions : ", self.cell_world_positions.shape)
        d_i = normalize_dr(self.cell_world_positions[:,:,dr.newaxis] - sources[:,dr.newaxis,:])[1]
        with open("results/d_i.txt", "w") as f:
            for i in range(d_i.shape[0]):
                f.write(f"{d_i[i, 0], d_i[i, 1]}\n")

        # [num_cells, num_modes]
        d_o = normalize_dr(self.cell_world_positions[:,:,dr.newaxis] - targets[:,dr.newaxis,:])[1]

        with open("results/d_o.txt", "w") as f:
            for i in range(d_o.shape[0]):
                f.write(f"{d_o[i, 0], d_o[i, 1]}\n")



        # Compute phases such that the total phase shifts for all cells
        # are equal 
        phases = self.cell_grid.wavenumber * (d_i+d_o)
        phases = dr.reshape(dr.cuda.ad.TensorXf, phases, (self.num_cols, self.num_rows, self.num_modes), order='A')
        self.phase_profile.values = phases

        # Set a neutral amplitude profile
        self.amplitude_profile.values = dr.ones(dr.cuda.ad.TensorXf, phases.shape)

        mode_powers = 1/dr.cuda.Float(self.num_modes) * \
            dr.ones(dr.cuda.ad.TensorXf, (self.num_modes,))
        self.amplitude_profile.mode_powers = mode_powers
        """

    def focusing_lens(self, sources, targets, remove_global_phase=True):
        """
        NOT WORKING YET
        Set RIS phase so that waves from 'sources' are focused onto 'targets' for each mode.

        sources: Tensor with shape (3, num_modes)
        targets: Tensor with shape (3, num_modes)
        self.cell_world_positions: Tensor with shape (3, num_rows*num_cols)
        """
        sources = dr.cuda.ad.TensorXf(sources)
        targets = dr.cuda.ad.TensorXf(targets)
        sources = expand_to_rank_dr(sources, 2, 0)  # (3, num_modes)
        targets = expand_to_rank_dr(targets, 2, 0)  # (3, num_modes)

        expected = (3, self.num_modes)
        for i, x in enumerate([sources, targets]):
            if not (x.shape[0] == expected[0] and x.shape[1] == expected[1]):
                raise ValueError(
                    f"Wrong shape of input {i+1}. Expected {expected}, got {x.shape}"
                )

        # Aliases
        Nm = self.num_modes
        Nc = self.num_rows * self.num_cols

        # cell positions: (3, Nc) -> (3, Nc, 1)
        cell_pos = self.cell_world_positions[:, :, dr.newaxis]
        
        # sources/targets: (3, Nm) -> (3, 1, Nm)
        src = sources[:, dr.newaxis, :]
        trg = targets[:, dr.newaxis, :]

        # Vectors from source->cell and cell->target: (3, Nc, Nm)
        v_in  = cell_pos - src
        v_out = trg - cell_pos  # (target - cell) has same length as (cell - target)

        # normalize_dr(vec) is assumed to return (unit_vec, length) along dimension 0
        _, d_in  = normalize_dr(v_in)   # (Nc, Nm)
        _, d_out = normalize_dr(v_out)  # (Nc, Nm)
        # Total optical path for each cell & mode
        total_path = d_in + d_out  # (Nc, Nm)
        # Convert path to phase shift: φ = -k * (d_in + d_out) modulo 2π
        k = self.cell_grid.wavenumber  # scalar

        raw_phase = k * total_path  # (Nc, Nm)
        # Wrap to [0, 2π)
        phases = raw_phase 
                
        phases = wrap_0_2pi(raw_phase)
        #phases = dr.select(phases < 0, phases + dr.two_pi, phases)  # (Nc, Nm)

        # Remove a per-mode global phase so pattern is anchored (optional)
        remove_global_phase = False
        if remove_global_phase:
            # Use cell 0 as reference; broadcast to all cells
            ref = phases[0, :]                        # (Nm,)
            phases = phases - ref[dr.newaxis, :]      # (Nc, Nm)
            phases = wrap_0_2pi(phases)#.__mod__(phases, dr.two_pi)
            phases = dr.select(phases < 0, phases + dr.two_pi, phases)

        # Reshape to (num_cols, num_rows, num_modes) as expected by the profiles
        # Current 'phases' is (Nc, Nm) with Nc = num_rows * num_cols.
        # If your grid is column-major (x fastest), map Nc -> (num_cols, num_rows).

        phases_reshaped = dr.empty(dr.cuda.ad.TensorXf, (self.num_rows, self.num_cols, Nm))
        for m in range(Nm):
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    phases_reshaped[r, c, m] = mi.Float(phases[r*self.num_cols + c, m])

        self.phase_profile.values = phases_reshaped#phases
        # Neutral amplitude
        self.amplitude_profile.values = dr.ones(dr.cuda.ad.TensorXf, phases_reshaped.shape)

        mode_powers = (1.0 / dr.cuda.Float(Nm)) * dr.ones(dr.cuda.ad.TensorXf, (Nm,))
        self.amplitude_profile.mode_powers = mode_powers

    def rotation(self) -> mi.Matrix3f:
        """
        World-from-local rotation using Z–Y–X (yaw–pitch–roll):
            R = Rz(alpha) * Ry(beta) * Rx(gamma)

        This returns a matrix that transforms local vectors to world:
            v_world = R @ v_local

        Note: If your eq. `rotation` uses a different order/axis convention,
        change the multiplication order accordingly.
        """
        alpha, beta, gamma = self.orientation.x, self.orientation.y, self.orientation.z
        return _rot_z(alpha) @ (_rot_y(beta) @ _rot_x(gamma))

    def __call__(self, points=None, mode=None, return_grads=False):
        # Fetch both modes at once (keeps things simple & fast)
        if return_grads and points is not None:
            phases_tx, grads_tx, hess_tx = self.phase_profile(points, None, True)
        else:
            phases_tx = self.phase_profile(points, None, False)

        a_tx = self.amplitude_profile(points, None)  # (M,B) TensorXf
        # Sanity: phases_tx.shape == a_tx.shape == (M,B)
        M = int(a_tx.shape[0])
        B = int(a_tx.shape[1])

        # 3) Mode weights -> sqrt(w)
        # mode_powers is length M (TensorXf), we want broadcast to (M,B)
        w_tx = dr.reshape(dr.cuda.ad.TensorXf, self.amplitude_profile.mode_powers, (M, 1))
        # elementwise sqrt, then broadcast to (M,B)
        sqrtw = dr.zeros(dr.cuda.ad.TensorXf, (M, B))
        for m in range(M):
            sm = dr.sqrt(dr.maximum(w_tx[m, 0], mi.Float(0.0)))
            sm = mi.Float(sm)
            sqrtw[m, :] = sm

        # 4) Build gamma = a * exp(i*phi) in real/imag (M,B)
        # use trigs directly on phases_tx (TensorXf)
        cphi = dr.cos(phases_tx)   # (M,B)
        sphi = dr.sin(phases_tx)   # (M,B)
        gamma_re = a_tx * cphi * sqrtw    # (M,B)
        gamma_im = a_tx * sphi * sqrtw    # (M,B)

        # 5) Select a single mode if requested
        if mode is not None:
            m = int(mode)
            if return_grads and points is not None:
                return (gamma_re[m, :], gamma_im[m, :]), grads_tx[m, :, :], hess_tx[m, :, :, :]
            else:
                return (gamma_re[m, :], gamma_im[m, :])

        # 6) Return all modes
        if return_grads and points is not None:
            return (gamma_re, gamma_im), grads_tx, hess_tx
        else:
            return (gamma_re, gamma_im)
        


def _rot_x(a: mi.Float) -> mi.Matrix3f:
    c, s = dr.cos(a), dr.sin(a)
    return mi.Matrix3f(
        1, 0, 0,
        0, c,-s,
        0, s, c
    )

def _rot_y(a: mi.Float) -> mi.Matrix3f:
    c, s = dr.cos(a), dr.sin(a)
    return mi.Matrix3f(
         c, 0, s,
         0, 1, 0,
        -s, 0, c
    )

def _rot_z(a: mi.Float) -> mi.Matrix3f:
    c, s = dr.cos(a), dr.sin(a)
    return mi.Matrix3f(
        c,-s, 0,
        s, c, 0,
        0, 0, 1
    )

def wrap_0_2pi(x): 
    # Returns x mod 2π in [0, 2π)
    two_pi = 2.0 * np.pi
    return x - two_pi * dr.floor(x / two_pi) 