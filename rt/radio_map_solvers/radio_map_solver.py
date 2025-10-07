#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: Copyright (c) 2025 Jean ACKER.
# SPDX-License-Identifier: Apache-2.0

"""Radio map solver"""

import mitsuba as mi
import drjit as dr
from typing import Tuple, Callable, List

from rt.utils.ray_tracing import spawn_ray_from_sources, fibonacci_lattice
from rt.utils.geometry import rotation_matrix
from rt.utils.misc import spectrum_to_matrix_4f

from rt.utils.irs_utils import mitsuba_rectangle_to_world, \
    gen_basis_from_z_dr, safe_rcp, outer3, I3,\
    gen_basis_from_k, compute_spreading_factor_dr

from sionna.rt import Scene
from sionna.rt.antenna_pattern import antenna_pattern_to_world_implicit
from sionna.rt.constants import InteractionType

from .radio_map import RadioMap


class RadioMapSolver:
    # pylint: disable=line-too-long
    r"""
    Class that implements the radio map solver

    This solver computes a radio map for every transmitter in the scene.
    For a given transmitter, a radio map is a rectangular surface with
    arbitrary orientation subdivded into rectangular cells of size
    :math:`\lvert C \rvert = \texttt{cell_size[0]} \times  \texttt{cell_size[1]}`.
    The parameter ``cell_size`` therefore controls the granularity of the
    map. The radio map associates with every cell :math:`(i,j)` the quantity

    .. math::
        :label: cm_def

        g_{i,j} = \frac{1}{\lvert C \rvert} \int_{C_{i,j}} \lvert h(s) \rvert^2 ds

    where :math:`\lvert h(s) \rvert^2` is the squared amplitude
    of the path coefficients :math:`a_i` at position :math:`s=(x,y)` assuming an
    ideal isotropic receiver,
    the integral is over the cell :math:`C_{i,j}`, and
    :math:`ds` is the infinitesimal small surface element
    :math:`ds=dx \cdot dy`.
    The dimension indexed by :math:`i` (:math:`j`) corresponds to the :math:`y\, (x)`-axis of the
    radio map in its local coordinate system. The quantity
    :math:`g_{i,j}` can be seen as the average :attr:`~sionna.rt.RadioMap.path_gain` across a cell.
    This solver computes an approximation of :math:`g_{i,j}` through Monte Carlo integration.

    The path gain can be transformed into the received signal strength (:attr:`~sionna.rt.RadioMap.rss`)
    by multiplying it with the transmit :attr:`~sionna.rt.Transmitter.power`:

    .. math::

        \mathrm{RSS}_{i,j} = P_{tx} g_{i,j}.

    If a scene has multiple transmitters, the
    signal-to-interference-plus-noise ratio
    (:attr:`~sionna.rt.Transmitter.sinr`) for transmitter :math:`k` is then
    defined as

    .. math::

        \mathrm{SINR}^k_{i,j}=\frac{\mathrm{RSS}^k_{i,j}}{N_0+\sum_{k'\ne k} \mathrm{RSS}^{k'}_{i,j}}

    where :math:`N_0` [W] is the :attr:`~sionna.rt.Scene.thermal_noise_power`, computed as:

    .. math::

        N_0 = B \times T \times k

    where :math:`B` [Hz] is the transmission :attr:`~sionna.rt.Scene.bandwidth`,
    :math:`T` [K] is the :attr:`~sionna.rt.Scene.temperature`, and
    :math:`k=1.380649\times 10^{-23}` [J/K] is the Boltzmann constant.

    The output of this function is a real-valued matrix of size ``[num_cells_y, num_cells_x]``,
    for every transmitter, with elements equal to the sum of the contributions of paths, and where

    .. math::
        \texttt{num_cells_x} = \bigg\lceil\frac{\texttt{size[0]}}{\texttt{cell_size[0]}} \bigg\rceil\\
        \texttt{num_cells_y} = \bigg\lceil \frac{\texttt{size[1]}}{\texttt{cell_size[1]}} \bigg\rceil.

    The surface defining the radio map is a rectangle centered at
    ``center``, with orientation ``orientation``, and with size
    ``size``. An orientation of (0,0,0) corresponds to
    a radio map parallel to the XY plane, with surface normal pointing towards
    the :math:`+z` axis. By default, the radio map
    is parallel to the XY plane, covers all of the scene, and has
    an elevation of :math:`z = 1.5\text{m}`.
    If transmitter and has multiple antennas, transmit precoding
    is applied which is defined by ``precoding_vec``.

    For every ray :math:`n` intersecting the radio map cell :math:`(i,j)`, the
    channel coefficients :math:`a_n` and the angles of departure (AoDs)
    :math:`(\theta_{\text{T},n}, \varphi_{\text{T},n})`
    and arrival (AoAs) :math:`(\theta_{\text{R},n}, \varphi_{\text{R},n})`
    are computed. See the `Primer on Electromagnetics <../em_primer.html>`_ for more details.

    A "synthetic" array is simulated by adding additional phase shifts that depend on the
    antenna position relative to the position of the transmitter as well as the AoDs.
    Let us denote by :math:`\mathbf{d}_{\text{T},k}\in\mathbb{R}^3` the relative position of
    the :math:`k^\text{th}` antenna (with respect to
    the position of the transmitter) for which the channel impulse response
    shall be computed. It can be accessed through the antenna array's property
    :attr:`~sionna.rt.AntennaArray.positions`. Using a plane-wave assumption, the resulting phase shift
    for this antenna can be computed as

    .. math::

        p_{\text{T},n,k} = \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},n}, \varphi_{\text{T},n})^\mathsf{T} \mathbf{d}_{\text{T},k}.

    The expression for the path coefficient of the :math:`k\text{th}` antenna is then

    .. math::

        h_{n,k} =  a_n e^{j p_{\text{T}, n,k}}.

    These coefficients form the complex-valued channel vector :math:`\mathbf{h}_n`
    of size :math:`\texttt{num_tx_ant}`.

    Finally, the coefficient of the equivalent SISO channel is

    .. math::
        h_n =  \mathbf{h}_n^{\textsf{H}} \mathbf{p}

    where :math:`\mathbf{p}` is the precoding vector ``precoding_vec``.

    Note
    -----
    This solver supports Russian roulette, which can significantly improve the
    efficiency of ray tracing by terminating rays that contribute little to the final
    result.

    The implementation of Russian roulette in this solver consists in terminating
    a ray with probability equal to the complement of its path gain (without
    the distance-based path loss). Formally,
    after the :math:`d^{\text{th}}` bounce, the ray path loss is set to:

    .. math::

        a_d \leftarrow
        \begin{cases}
            \frac{a_d}{\sqrt{\min \{ p_{c},|a_d|^2 \}}},  & \text{with probability } \min \{ p_{c},|a_d|^2 \}\\
            0, & \text{with probability } 1 - \min \{ p_{c},|a_d|^2 \}
        \end{cases}

    where :math:`a_d` is the path coefficient corresponding to the ray (without
    the distance-based pathloss) and
    :math:`p_c` the maximum probability with which to continue a path (``rr_prob``).
    The first case consists in continuing the ray, whereas the second case consists
    in terminating the ray. When the ray is continued, the scaling by
    :math:`\frac{1}{\sqrt{\min \{ p_{c},|a_d|^2 \}}}` ensures an unbiased map by accounting
    for the rays that were terminated. When a ray is terminated, it is no longer
    traced, leading to a reduction of the required computations.

    Russian roulette is by default disabled. It can be enabled by setting
    the ``rr_depth`` parameter to a positive value. ``rr_depth`` corresponds to
    the path depth, i.e., the number of bounces, from which on Russian roulette
    is enabled.

    Note
    -----
    The parameter ``stop_threshold`` can be set to deactivate (i.e., stop tracing)
    paths whose gain has dropped below this threshold (in dB).

    Example
    -------
    .. code-block:: Python

        import sionna
        from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

        scene = load_scene(sionna.rt.scene.munich)
        scene.radio_materials["marble"].thickness = 0.5

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=8,
                                num_cols=2,
                                vertical_spacing=0.7,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="VH")

        # Add a transmitters
        tx = Transmitter(name="tx",
                    position=[8.5,21,30],
                    orientation=[0,0,0])
        scene.add(tx)
        tx.look_at(mi.Point3f(40,80,1.5))

        solver = RadioMapSolver()
        rm = solver(scene, cell_size=(1., 1.), samples_per_tx=100000000)
        scene.preview(radio_map=rm, clip_at=15., rm_vmin=-100.)

    .. figure:: ../figures/radio_map_preview.png
        :align: center
    """

    DISCARD_THRES = 1e-15  # -150 dB, matches old SolverCoverageMap
    def __init__(self):
        # Sampler
        self._sampler = mi.load_dict({'type': 'independent'})
        # Dr.Jit mode for running the loop that implement the solver.
        # Symbolic mode is the fastest mode but does not currently support
        # automatic differentiation.
        self._loop_mode = "symbolic"

    @property
    def loop_mode(self):
        # pylint: disable=line-too-long
        r"""Get/set the Dr.Jit mode used to evaluate the loop that implements
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.
        For more details, see the `corresponding Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/cflow.html#sym-eval>`_.

        :type: "evaluated" | "symbolic"
        """
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._loop_mode = mode

    def _build_mi_ris_objects(self, scene):
        r"""
        Builds a Mitsuba scene containing all RIS as rectangles with position,
        orientation, and size matching the RIS properties.

        Output
        ------
        : list(mi.Rectangle)
            List of Mitsuba rectangles implementing the RIS

        : mi.UInt
            RIS indices
        """
        # List of all the RIS objects in the scene
        all_ris = list(scene.irs.values())
        num_ris = len(all_ris)

        # Creates a scene containing RIS as rectangles
        mi_ris_objects = []
        mi_ris_ids_u32 = [] 
        for i, ris in enumerate(all_ris):
            center = ris.position
            orientation = ris.orientation
            size = ris.size
            mi_to_world = mitsuba_rectangle_to_world(center, orientation, size,
                                                     ris=True)
            ris_rect = mi.load_dict({   "type"     : "rectangle",
                                        "to_world" : mi_to_world
                                    })
            mi_ris_objects.append(ris_rect)
            rid = dr.reinterpret_array(mi.UInt32, mi.ShapePtr(ris_rect))[0]
            mi_ris_ids_u32.append(rid)
        return mi_ris_objects, mi_ris_ids_u32

    def __call__(
        self,
        scene : Scene,
        center : mi.Point3f | None = None,
        orientation : mi.Point3f | None = None,
        size : mi.Point2f | None = None,
        cell_size : mi.Point2f = mi.Point2f(10, 10),
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf] | None = None,
        samples_per_tx : int = 1000000,
        max_depth : int = 3,
        los : bool = True,
        specular_reflection : bool = True,
        diffuse_reflection : bool = False,
        refraction : bool = True,
        seed : int = 42,
        rr_depth : int = -1,
        rr_prob : float = 0.95,
        stop_threshold : float | None = None
        ) -> RadioMap:
        # pylint: disable=line-too-long
        r"""
        Executes the solver

        :param scene: Scene for which to compute the radio map

        :param center: Center of the radio map :math:`(x,y,z)` [m] as
            three-dimensional vector. If set to `None`, the radio map is
            centered on the center of the scene, except for the elevation
            :math:`z` that is set to 1.5m. Otherwise, ``orientation`` and
            ``size`` must be provided.

        :param orientation: Orientation of the radio map
            :math:`(\alpha, \beta, \gamma)` specified through three angles
            corresponding to a 3D rotation as defined in :eq:`rotation`.
            An orientation of :math:`(0,0,0)` or `None` corresponds to a
            radio map that is parallel to the XY plane.
            If not set to `None`, then ``center`` and ``size`` must be
            provided.

        :param size:  Size of the radio map [m]. If set to `None`, then the
            size of the radio map is set such that it covers the entire scene.
            Otherwise, ``center`` and ``orientation`` must be provided.

        :param cell_size: Size of a cell of the radio map [m]

        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector.
            If set to `None`, then defaults to
            :math:`\frac{1}{\sqrt{\text{num_tx_ant}}} [1,\dots,1]^{\mathsf{T}}`.

        :param samples_per_tx: Number of samples per source

        :param max_depth: Maximum depth

        :param los: Enable line-of-sight paths

        :param specular_reflection: Enable specularl reflections

        :param diffuse_reflection: Enable diffuse reflectios

        :param refraction: Enable refraction

        :param seed: Seed

        :param rr_depth: Depth from which on to start Russian roulette

        :param rr_prob: Maximum probability with which to keep a path when
            Russian roulette is enabled
        :param stop_threshold: Gain threshold [dB] below which a path is
            deactivated

        :return: Computed radio map
        """

        # Check that the scene is all set for simulations
        scene.all_set(radio_map=True)

        # Check the properties of the rectangle defining the radio map
        if ((center is None) and (size is None) and (orientation is None)):
            # Default value for center: Center of the scene
            # Default value for the scale: Just enough to cover all the scene
            # with axis-aligned edges of the rectangle
            # [min_x, min_y, min_z]
            scene_min = scene.mi_scene.bbox().min
            # In case of empty scene, bbox min is -inf
            scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)
            # [max_x, max_y, max_z]
            scene_max = scene.mi_scene.bbox().max
            # In case of empty scene, bbox min is inf
            scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)
            # Center and size
            center = 0.5 * (scene_min + scene_max)
            center.z = 1.5
            size = scene_max - scene_min
            size = mi.Point2f(size.x, size.y)
            # Set the orientation to default value
            orientation = dr.zeros(mi.Point3f, 1)
        elif ((center is None) or (size is None) or (orientation is None)):
            raise ValueError("If one of `cm_center`, `cm_orientation`,"\
                             " or `cm_size` is not None, then all of them"\
                             " must not be None")
        else:
            center = mi.Point3f(center)
            orientation = mi.Point3f(orientation)
            size = mi.Point2f(size)

        # Check and initialize the precoding vector
        num_tx = len(scene.transmitters)
        num_tx_ant = scene.tx_array.num_ant
        if precoding_vec is None:
            precoding_vec_real = dr.ones(mi.TensorXf, [num_tx, num_tx_ant])
            precoding_vec_real /= dr.sqrt(scene.tx_array.num_ant)
            precoding_vec_imag = dr.zeros(mi.TensorXf, [num_tx, num_tx_ant])
            precoding_vec = (precoding_vec_real, precoding_vec_imag)
        else:
            precoding_vec_real, precoding_vec_imag = precoding_vec
            if not isinstance(precoding_vec_real, type(precoding_vec_imag)):
                raise TypeError("The real and imaginary components of "\
                                "`precoding_vec` must be of the same type")
            # If a single precoding vector is provided, then it is used by
            # all transmitters
            if ( isinstance(precoding_vec_real, mi.Float) or
                (isinstance(precoding_vec_real, mi.TensorXf)
                 and len(dr.shape(precoding_vec_real)) == 1) ):
                precoding_vec_real = mi.Float(precoding_vec_real)
                precoding_vec_imag = mi.Float(precoding_vec_imag)

                precoding_vec_real = dr.tile(precoding_vec_real, num_tx)
                precoding_vec_imag = dr.tile(precoding_vec_imag, num_tx)
                #
                precoding_vec_real = dr.reshape(mi.TensorXf, precoding_vec_real,
                                                [num_tx, num_tx_ant])
                precoding_vec_imag = dr.reshape(mi.TensorXf, precoding_vec_imag,
                                                [num_tx, num_tx_ant])
                precoding_vec = (precoding_vec_real, precoding_vec_imag)

        # Builds the Mitsuba scene with RIS for
        # testing intersections with RIS
        mi_ris_objects, mi_ris_ids_u32 = self._build_mi_ris_objects(scene=scene)
        
        # Transmitter configurations
        # Generates sources positions and orientations
        tx_positions, tx_orientations, rel_ant_positions_tx, _ =\
                                                    scene.sources(True, False)
        dr.make_opaque(tx_positions, tx_orientations)

        # Trace paths and compute channel impulse responses
        tx_antenna_patterns = scene.tx_array.antenna_pattern.patterns

        num_tx = dr.shape(tx_positions)[1]
        num_samples = samples_per_tx*num_tx

        # If the Russian roulette threshold depth is set to -1, disable Russian
        # roulette by setting the threshold depth to a value higher than
        # `max_depth`
        if rr_depth == -1:
            rr_depth = max_depth + 1

        # If a threshold for the path gain is set below which paths are
        # deactivated, then convert it to linear scale
        if stop_threshold is not None:
            stop_threshold = dr.power(10., stop_threshold/10.)

        # Set the seed of the sampler
        self._sampler.seed(seed, num_samples)

        # Allocate the pathloss map
        radio_map = RadioMap(scene, center, orientation, size, cell_size)

        # Computes the pathloss map
        # `radio_map` is updated in-place
        ris = True if len(mi_ris_objects) > 0 else False
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            self._shoot_and_bounce(scene, radio_map, mi_ris_objects, mi_ris_ids_u32, self._sampler,
                                tx_positions, tx_orientations, tx_antenna_patterns,
                                precoding_vec, rel_ant_positions_tx,
                                samples_per_tx, max_depth,
                                los, specular_reflection, diffuse_reflection,ris,
                                refraction, rr_depth, rr_prob,
                                stop_threshold)

        return radio_map

    # pylint: disable=line-too-long
     #@dr.syntax
    def _shoot_and_bounce(
        self,
        scene : Scene,
        radio_map : RadioMap,
        mi_ris_objects,
        mi_ris_ids_u32: List[int],
        sampler : mi.Sampler,
        tx_positions : mi.Point3f,
        tx_orientations : mi.Point3f,
        tx_antenna_patterns : List[Callable[[mi.Float, mi.Float],
                                            Tuple[mi.Complex2f, mi.Complex2f]]],
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf],
        rel_ant_positions_tx : mi.Point3f,
        samples_per_tx : int,
        max_depth : int,
        los : bool,
        specular_reflection : bool,
        diffuse_reflection : bool,
        ris : bool,
        refraction : bool,
        rr_depth : int,
        rr_prob : float,
        stop_threshold : float | None,
    ) -> None:
        r"""
        Executes the shoot-and-bounce loop

        The ``radio_map`` is updated in-place.

        :param scene: Scene for which to compute the radio map
        :param radio_map: Radio map
        :param sampler: Sampler used to generate random numbers and vectors
        :param tx_positions: Positions of the transmitters
        :param tx_orientations: Orientations of the transmitters
        :param tx_antenna_patterns: Antenna pattern of the transmitters
        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param samples_per_tx: Number of samples per source
        :param max_depth: Maximum depth
        :param los: If set to `True`, then the line-of-sight paths are computed
        :param specular_reflection: If set to `True`, then the specularly
            reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely
            reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are
            computed
        :param rr_depth: Depth from which to start Russian roulette
        :param rr_prob: Minimum probability with which to keep a path when
            Russian roulette is enabled
        :param stop_threshold: Gain threshold (linear scale) below which a path
            is deactivated
        """
        # ---------- setup ----------
        num_txs = dr.shape(tx_positions)[1]
        num_samples = samples_per_tx * num_txs
        num_tx_ant_patterns = len(tx_antenna_patterns)

        # Per-ray transmitter index: [0..0..,1..1.., ...] block-wise
        #tx_indices = dr.arange(mi.UInt32, num_samples) // mi.UInt32(samples_per_tx)
        tx_indices = dr.repeat(dr.arange(mi.UInt, num_txs), samples_per_tx)

        # Spawn rays from sources (returns initial ray + k_tx directions)
        ray = spawn_ray_from_sources(fibonacci_lattice, samples_per_tx, tx_positions)
        k_tx = ray.d
        # Array weighting (synth array + precoding)
        array_w = self._synthetic_array_weighting(scene, ray.d, rel_ant_positions_tx, precoding_vec)

        # Active mask and tube bookkeeping
        active = dr.full(dr.mask_t(mi.Float), True, num_samples)
        if stop_threshold is not None:
            ray_tube_length = dr.zeros(mi.Float, num_samples)
        else:
            ray_tube_length = None

        # Initial solid angle per ray (uniform over sphere per TX)
        solid_angle = dr.full(mi.Float, 4.0 * dr.pi * dr.rcp(samples_per_tx),
                              num_samples)
        # Per-ray TX orientation via gather on tx_indices
        sample_tx_orientation = dr.repeat(tx_orientations, samples_per_tx)
        tx_to_world = rotation_matrix(sample_tx_orientation)
        # Initialize electric field in world-implicit basis
        e_fields = [
            antenna_pattern_to_world_implicit(src_antenna_pattern, tx_to_world, ray.d, direction="out")
            for src_antenna_pattern in tx_antenna_patterns
        ]
        # RIS support
        if ris:
            # Radii = [R1, R2] and principal directions U,V ⟂ k
            radii_curv = dr.zeros(mi.Vector2f, num_samples)
            dirs_curv_u, dirs_curv_v = gen_basis_from_z_dr(k_tx, eps=1e-6)

        else:
            radii_curv = None
            dirs_curv_u = None
            dirs_curv_v = None
            mi_ris_objects = None

        depth = mi.UInt32(0)
        #dr.set_flag(dr.JitFlag.Symbolic, False)
        # ---------- path loop ----------
        #prev = dr.get_flag(dr.JitFlag.SymbolicLoops)
        #dr.set_flag(dr.JitFlag.SymbolicLoops, False)   # evaluated mode
        #try:
        #while bool(dr.any(active)): 
        while dr.hint(dr.any(active), mode=self.loop_mode, exclude=[array_w]):
            # Scene intersection
            si_scene = scene.mi_scene.ray_intersect(ray, active=active)

            # Measurement plane intersection (for radio map)
            si_mp = radio_map.measurement_plane.ray_intersect(ray, active=active)

            # RIS intersection (against separate accel/scene)
            if ris:
                hit_ris, t_ris, ris_ind = self._ris_intersect(mi_ris_objects, ray, True)
            else:
                hit_ris = dr.full(mi.Bool, False, num_samples)
                t_ris   = dr.full(mi.Float, dr.inf, num_samples)
                ris_ind = dr.full(mi.Int32, -1, num_samples)

            # Which surface comes first (scene vs RIS)?
            hit_scene = si_scene.is_valid() & (si_scene.t < t_ris) 
            hit_ris   = hit_ris & ((t_ris <= si_scene.t) | ~si_scene.is_valid())

            # Active if a surface is hit at all
            active = hit_scene | hit_ris

            # Measurement-plane hit must come before any surface hit and be valid
            val_mp_int = si_mp.is_valid() & (si_mp.t < si_scene.t) & (si_mp.t < t_ris)
            # Disable LoS if requested
            val_mp_int &= (depth > 0) | los

            # Update the radio map
            radio_map.add(
                e_fields=e_fields,
                solid_angle=solid_angle,
                array_w=array_w,
                si_mp=si_mp,
                k_world=ray.d,
                ray_origin=ray.o,             # if your add() needs it for RIS case
                tx_indices=tx_indices,
                hit=val_mp_int,
                ris=ris,
                radii_curv=radii_curv,        # used if you integrated ray-tube weighting (RIS)
            )

            # ---- SCENE continuation (BSDF) ----
            sample1 = self._sampler.next_1d()
            sample2 = self._sampler.next_2d()
            e_fields_scene = dr.copy(e_fields)
            s, e_fields_scene = self._sample_radio_material(
                si=si_scene, k_world=ray.d, e_fields=e_fields_scene,
                solid_angle=solid_angle, sample1=sample1, sample2=sample2,
                specular_reflection=specular_reflection,
                diffuse_reflection=diffuse_reflection,
                refraction=refraction,
                active=hit_scene  # only valid where scene was hit
            )

            interaction_type = dr.select(hit_scene, s.sampled_component, InteractionType.NONE)
            is_diffuse = hit_scene & (interaction_type == InteractionType.DIFFUSE)

            # Solid angle only changes for diffuse reflection
            solid_angle_scene = dr.select(is_diffuse, dr.two_pi, solid_angle)
            ray_scene = si_scene.spawn_ray(d=s.wo)


            # Keep curvature as-is for scene path
            radii_scene   = radii_curv
            dirs_u_scene  = dirs_curv_u
            dirs_v_scene  = dirs_curv_v

            # ---- RIS continuation ----
            P_ris = ray.o + t_ris * ray.d
            sentinel = mi.UInt32(0xFFFFFFFF)

            ris_ind_eff = dr.select(hit_ris, ris_ind, sentinel)
            
            if ris:
                e_fields_ris, k_r, R_ris, U_ris, V_ris, n_ris = self._compute_ris_reflected_field(
                    scene, 
                    mi_ris_ids_u32=mi_ris_ids_u32,
                    int_point   = P_ris,
                    ris_ind     = ris_ind_eff,
                    k_i         = ray.d,
                    e_fields    = e_fields,
                    length      = t_ris,
                    radii_curv  = radii_curv,
                    dirs_curv_u = dirs_curv_u,
                    dirs_curv_v = dirs_curv_v
                )
                eps = mi.Float(1e-4)
                o_ris = P_ris + eps * n_ris
                d_ris = k_r

            else:
                # Dummy values (won’t be used because hit_ris is false everywhere)
                e_fields_ris = e_fields
                o_ris = ray.o
                d_ris = ray.d
                R_ris = radii_curv
                U_ris = dirs_curv_u
                V_ris = dirs_curv_v

            # ---- Merge SCENE vs RIS (per-lane) ----
            ray_next = ray_scene
            ray_next.o = dr.select(hit_ris, o_ris, ray_next.o)
            ray_next.d = dr.select(hit_ris, d_ris, ray_next.d)

            # Fields
            Pn = len(e_fields)
            e_fields = [dr.select(hit_ris, e_fields_ris[p], e_fields_scene[p]) for p in range(Pn)]

            # Solid angle: keep previous for RIS lanes (delta-like), update for scene lanes
            solid_angle = dr.select(hit_ris, solid_angle, solid_angle_scene)

            # Curvature / principal dirs
            
            if ris:
                radii_curv  = dr.select(hit_ris, R_ris,  radii_scene)
                dirs_curv_u = dr.select(hit_ris, U_ris,  dirs_u_scene)
                dirs_curv_v = dr.select(hit_ris, V_ris,  dirs_v_scene)
            
            # Previous intersection point (useful if your radio_map needs it elsewhere)
            prev_int_point_scene = si_scene.p
            ray = ray_next

            # ---- Depth / continuation ----
            depth += 1
            # Continue if we (a) took a scene bounce that actually sampled a component, or (b) hit a RIS
            cont = (hit_scene & (interaction_type != InteractionType.NONE)) | hit_ris
            active &= cont
            active &= (depth <= max_depth)

            # ---- Russian roulette & stop-by-gain ----
            # Path gain (no spreading)
            gain = dr.zeros(mi.Float, 1)
            for ef in e_fields:
                gain += dr.squared_norm(ef)
            gain /= num_tx_ant_patterns

            if stop_threshold is not None:
                # Segment length used in this step
                seg_t = dr.select(hit_ris, t_ris, si_scene.t)
                # Reset tube length for diffuse bounces (Lambertian re-emission)
                ray_tube_length = dr.select(is_diffuse, 0.0, ray_tube_length)
                ray_tube_length += seg_t

            # RR activation after rr_depth
            rr_inactive = (depth < rr_depth)
            rr_continue_prob = dr.minimum(gain, rr_prob)
            rr_continue = self._sampler.next_1d() < rr_continue_prob
            active &= (rr_inactive | rr_continue)

            # Unbiasedness scaling
            for i in range(num_tx_ant_patterns):
                e_fields[i] = dr.select(rr_inactive, e_fields[i],
                                        e_fields[i] * dr.rsqrt(rr_continue_prob))

            # Stop by path gain threshold (after including spreading)
            if stop_threshold is not None:
                gain_pl = gain * dr.sqr(scene.wavelength * dr.rcp(4.0 * dr.pi * ray_tube_length))
                active &= (gain_pl > stop_threshold)

    
        # finalize
        radio_map.finalize()
        #finally:
        #    dr.set_flag(dr.JitFlag.SymbolicLoops, True)  # restore
        #    dr.set_flag(dr.JitFlag.Debug, True)
    
    def _ris_intersect(self, ris_objects, ray, active):
        r"""
        Test the intersection with the RIS

        Input
        ------
        ris_objects : list(mi.Rectangle)
            List of Mitsuba rectangles implementing the RIS

        Output
        -------
        valid : mi.Bool
            Mask indicating if the intersection is valid

        t : mi.Float
            Position of the intersection on the ray

        indices : mi.UInt32
            Indices of the intersected RIS
        """

        num_rays = dr.shape(ray.d)[1]
        t = dr.full(mi.Float, float('inf'), num_rays)
        valid = dr.full(mi.Bool, False, num_rays)
        indices = dr.full(mi.UInt, 0, num_rays)
        for ris in ris_objects:
            si_ris = ris.ray_intersect(ray, active=active)
            v_ = si_ris.is_valid()
            t_ = si_ris.t
            indices_ = dr.reinterpret_array(mi.UInt32, si_ris.shape)

            valid |= v_
            new_closest = v_ & (t_ < t)

            t = dr.select(new_closest, t_, t)
            indices = dr.select(new_closest, indices_, indices)

        return valid, t, indices
    
    def _compute_ris_reflected_field(
        self,
        scene: Scene,
        mi_ris_ids_u32: List[int],
        int_point: mi.Point3f,            # [N] intersection points with RIS (world)
        ris_ind: mi.Int32,                 # [N] RIS object id per ray
        k_i: mi.Vector3f,                  # [N] incident unit directions (world)
        e_fields: List[mi.Vector4f],       # list length P, each [N]-wide Vector4f
        length: mi.Float,                  # [N] last segment length
        radii_curv: mi.Vector2f,           # [N] [R1,R2]
        dirs_curv_u: mi.Vector3f,          # [N] principal dir #1 (world)
        dirs_curv_v: mi.Vector3f           # [N] principal dir #2 (world)
    ) -> Tuple[List[mi.Vector4f], mi.Vector3f, mi.Vector2f, mi.Vector3f, mi.Vector3f, mi.Vector3f]:
        """
        DR.Jit rewrite: compute field & geometry after RIS reflection.
        Returns:
            e_fields_out : list[mi.Vector4f]  (updated world-implicit fields)
            k_r          : mi.Vector3f        (reflected directions)
            radii_out    : mi.Vector2f        (updated radii)
            dirs_u_out   : mi.Vector3f        (updated principal dir #1)
            dirs_v_out   : mi.Vector3f        (updated principal dir #2)
            normal_out   : mi.Vector3f        (RIS normal at hit; 0 where not RIS)
        """

        P = len(e_fields)

        # 0) Apply pre-RIS spreading factor (advance from previous point to RIS)
        #    This matches old TF code: scale field, then R += length
        sf = compute_spreading_factor_dr(radii_curv.x, radii_curv.y, length)  # [N]

        for p in range(P):
            e_fields[p] *= sf
        radii_curv = radii_curv + length  # broadcasts to both components

        # Prepare outputs as copies; we will mask-update lanes that actually hit a RIS front side
        e_out = [ef for ef in e_fields]
        k_r   = mi.Vector3f(0.0)               # default (inactive lanes)
        n_out = mi.Vector3f(0.0)
        R_out = radii_curv
        U_out = dirs_curv_u
        V_out = dirs_curv_v

        # Iterate all RIS objects once; update rays hitting that RIS (front side)
        # Assumptions on `self._scene.ris.values()`:
        #   - each `ris` has: `object_id: int`, `position: mi.Point3f`,
        #                     `rotation: mi.Matrix3f` (world-from-local),
        #                     `world_normal: mi.Vector3f` (front normal),
        #                     `amplitude_profile.mode_powers: 1D array-like`
        #   - callable `ris(local_uv, return_grads=True)` yielding:
        #       gamma_m  : [M, N_ris]    (mode amplitude per mode m)
        #       grad_m   : [M, N_ris, 3] (phase gradient in LOCAL frame)
        #       hess_m   : [M, N_ris, 3,3] (phase Hessian in LOCAL frame)
        # If your API differs, adapt the commented blocks accordingly.

        # Wavenumber k0 = 2*pi / lambda
        k0 = scene.wavenumber

        for ris, rid_u32 in zip(scene.irs.values(), mi_ris_ids_u32):
            # per-RIS mask (add & hit_ris if you maintain it)
            m_ris = (rid_u32 == ris_ind)  # & hit_ris
            n     = ris.world_normal
            m_front = m_ris & (dr.dot(k_i, n) < 0.0)
            # --- world -> local (RIS) coords (full width) ---
            pos  = ris.position  
            R_wl = dr.transpose(ris.rotation())
            P_l  = R_wl @ (int_point - pos)

            uv   = mi.Vector2f(P_l.y, P_l.z)
            # --- RIS spatial modulation ---
            (gamma_m_re, gamma_m_im), grad_m_L, hess_m_L = ris(uv, return_grads=True)
            M = int(gamma_m_re.shape[0])

            w_host = getattr(ris, "mode_powers_host", None)
            if w_host is None:
                # uniform fallback when no host list is available
                w_host = [1.0] * M

            s = sum(w_host) or float(M)
            cdf = []
            acc = 0.0
            for wi in w_host:
                acc += wi / s
                cdf.append(min(acc, 1.0))

            u    = self._sampler.next_1d()              # mi.Float [B]
            mode = mi.Int32(M - 1)
            for j, cj in enumerate(cdf):
                mode = dr.select(u < mi.Float(cj), mi.Int32(j), mode)

            # -------- select per-mode quantities from packed tensors --------
            # gamma: select re/im separately, then magnitude
            re_sel = mi.Float(0.0)
            im_sel = mi.Float(0.0)
            for j in range(M):
                is_j = (mode == j)
                re_j = gamma_m_re[j, :].array          # mi.Float[B]
                im_j = gamma_m_im[j, :].array
                re_sel = dr.select(is_j, re_j, re_sel)
                im_sel = dr.select(is_j, im_j, im_sel)

            # grad_l: Vector3f per lane
            gx = mi.Float(0.0); gy = mi.Float(0.0); gz = mi.Float(0.0)
            for j in range(M):
                is_j = (mode == j)
                gj0 = grad_m_L[j, :, 0].array          # mi.Float[B]
                gj1 = grad_m_L[j, :, 1].array
                gj2 = grad_m_L[j, :, 2].array
                gx = dr.select(is_j, gj0, gx)
                gy = dr.select(is_j, gj1, gy)
                gz = dr.select(is_j, gj2, gz)
            grad_l = mi.Vector3f(gx, gy, gz)

            # Ql: Matrix3f per lane
            Qxx = mi.Float(0.0); Qxy = mi.Float(0.0); Qxz = mi.Float(0.0)
            Qyx = mi.Float(0.0); Qyy = mi.Float(0.0); Qyz = mi.Float(0.0)
            Qzx = mi.Float(0.0); Qzy = mi.Float(0.0); Qzz = mi.Float(0.0)
            for j in range(M):
                is_j = (mode == j)
                H = hess_m_L  # alias
                Qxx = dr.select(is_j, H[j, :, 0, 0].array, Qxx)
                Qxy = dr.select(is_j, H[j, :, 0, 1].array, Qxy)
                Qxz = dr.select(is_j, H[j, :, 0, 2].array, Qxz)
                Qyx = dr.select(is_j, H[j, :, 1, 0].array, Qyx)
                Qyy = dr.select(is_j, H[j, :, 1, 1].array, Qyy)
                Qyz = dr.select(is_j, H[j, :, 1, 2].array, Qyz)
                Qzx = dr.select(is_j, H[j, :, 2, 0].array, Qzx)
                Qzy = dr.select(is_j, H[j, :, 2, 1].array, Qzy)
                Qzz = dr.select(is_j, H[j, :, 2, 2].array, Qzz)
            Ql = mi.Matrix3f([[Qxx, Qxy, Qxz],
                            [Qyx, Qyy, Qyz],
                            [Qzx, Qzy, Qzz]])

            # -------- to WORLD --------
            R_w = ris.rotation()                  # Matrix3f (constant)
            grad_w = R_w @ grad_l
            Qw     = R_w @ (Ql @ dr.transpose(R_w))

            # -------- reflected direction (phase-gradient steering) --------
            proj     = dr.dot(n, k_i)
            tang     = k_i - n * proj
            grad_i   = (-k0) * tang
            grad     = grad_i + grad_w

            kr_t     = -(grad * safe_rcp(k0))
            kr_t_n2  = dr.squared_norm(kr_t)
            kr_n_mag = dr.sqrt(dr.maximum(0.0, 1.0 - kr_t_n2))
            Kr       = dr.normalize(kr_t + kr_n_mag * n)


            # -------- curvature transport --------
            invR1 = safe_rcp(R_out.x); invR2 = safe_rcp(R_out.y)
            Q_i   = invR1 * outer3(U_out, U_out) + invR2 * outer3(V_out, V_out)
            denom = dr.dot(Kr, n)
            L     = I3() - outer3(Kr, n) * safe_rcp(denom)
            Q_r   = dr.transpose(L) @ (Q_i - (Qw * (1.0 / k0))) @ L  # keep if needed

            # -------- update basis (approx) --------
            U_new, V_new = gen_basis_from_k(Kr)
            R_new        = R_out

            # -------- apply amplitude (|gamma|) to fields --------
            amp = dr.sqrt(re_sel * re_sel + im_sel * im_sel)   # magnitude per lane

            for p in range(P):
                e_out[p] = dr.select(m_front, e_out[p] * amp, e_out[p])

            # -------- gated lane-wise updates --------
            k_r   = dr.select(m_front, Kr,    k_r)
            n_out = dr.select(m_front, n,     n_out)
            R_out = dr.select(m_front, R_new, R_out)
            U_out = dr.select(m_front, U_new, U_out)
            V_out = dr.select(m_front, V_new, V_out)

            return e_out, k_r, R_out, U_out, V_out, n_out
        
    @dr.syntax
    def _synthetic_array_weighting(
        self,
        scene : Scene,
        k_tx : mi.Vector3f,
        rel_ant_positions_tx : mi.Point3f,
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf],
        ) -> List[mi.Float]:
        r"""
        Computes the weighting to apply to the electric field to synthetically
        model the transmitter array

        :param scene: Scene for which to compute the radio map
        :param k_tx: Directions of departures of paths
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector

        :return: Weightings
        """

        precoding_vec_real, precoding_vec_imag = precoding_vec
        array_size = scene.tx_array.array_size
        num_patterns = len(scene.tx_array.antenna_pattern.patterns)
        num_tx = len(scene.transmitters)
        samples_per_tx = dr.shape(k_tx)[-1] // num_tx

        # Reshape to split transmitters and array samples per tx
        # Note: Split the x,y,z coordinates to handle large number of samples,
        # as the maximum size allowed for one array is 2^32
        # [num_tx, samples_per_tx]
        k_tx_x = dr.reshape(mi.TensorXf, k_tx.x, [num_tx, samples_per_tx])
        k_tx_y = dr.reshape(mi.TensorXf, k_tx.y, [num_tx, samples_per_tx])
        k_tx_z = dr.reshape(mi.TensorXf, k_tx.z, [num_tx, samples_per_tx])

        # Reshape relative antenna positions
        # Add a dimension to broadcast with samples per tx
        # Note: Split the x,y,z coordinates to handle large number of samples,
        # as the maximum size allowed for one array is 2^32
        # [num_tx, 1, array_size]
        rel_ant_positions_tx_x = dr.reshape(mi.TensorXf, rel_ant_positions_tx.x,
                                            [num_tx, 1, array_size])
        rel_ant_positions_tx_y = dr.reshape(mi.TensorXf, rel_ant_positions_tx.y,
                                            [num_tx, 1, array_size])
        rel_ant_positions_tx_z = dr.reshape(mi.TensorXf, rel_ant_positions_tx.z,
                                            [num_tx, 1, array_size])

        # Reshape precoding vector
        # Add dimension to broadcast with number of samples, and split patterns
        # from array
        # [num_tx, 1, num_patterns, array_size]
        precoding_vec_real_ = dr.reshape(mi.TensorXf, precoding_vec_real.array,
                                         [num_tx, 1, num_patterns, array_size])
        precoding_vec_imag_ = dr.reshape(mi.TensorXf, precoding_vec_imag.array,
                                         [num_tx, 1, num_patterns, array_size])
        precoding_vec_real = []
        precoding_vec_imag = []
        # [num_tx, 1, array_size]
        for i in range(num_patterns):
            precoding_vec_real.append(precoding_vec_real_[...,i,:])
            precoding_vec_imag.append(precoding_vec_imag_[...,i,:])

        # To reduce the memory footprint, iterate over the number of transmitter
        # antennas. This avoids allocating tensors with shape
        #                                           [num_samples, num_ant, ...]
        # Weights for each antenna pattern
        w_real = []
        w_imag = []
        for i in range(num_patterns):
            w_real.append(dr.zeros(mi.TensorXf, [num_tx, samples_per_tx]))
            w_imag.append(dr.zeros(mi.TensorXf, [num_tx, samples_per_tx]))
        n = 0
        while n < array_size:
            # Extract the relative position of antenna
            # [num_tx, 1]
            ant_pos_x = rel_ant_positions_tx_x[...,n]
            ant_pos_y = rel_ant_positions_tx_y[...,n]
            ant_pos_z = rel_ant_positions_tx_z[...,n]
            # [num_tx, samples_per_tx]
            tx_phase_shifts = ant_pos_x*k_tx_x + ant_pos_y*k_tx_y\
                                + ant_pos_z*k_tx_z
            tx_phase_shifts *= dr.two_pi/scene.wavelength
            array_vec_imag, array_vec_real = dr.sincos(tx_phase_shifts)
            for i in range(num_patterns):
                # Dot product with precoding vector iteratively computed
                # [num_tx, 1]
                prec_real = precoding_vec_real[i][...,n]
                prec_imag = precoding_vec_imag[i][...,n]
                # [num_tx, samples_per_tx, num_patterns]
                w_real[i] += array_vec_real * prec_real\
                            - array_vec_imag * prec_imag
                w_imag[i] += array_vec_real * prec_imag\
                            + array_vec_imag * prec_real
            #
            n += 1

        # Reshape to fit total number of samples
        w = []
        for i in range(num_patterns):
            w_real_ = dr.reshape(mi.Float, w_real[i], [num_tx*samples_per_tx])
            w_imag_ = dr.reshape(mi.Float, w_imag[i], [num_tx*samples_per_tx])
            w_ = mi.Matrix4f(w_real_,     0.0,    -w_imag_,        0.0,
                             0.0,     w_real_,        0.0,    -w_imag_,
                             w_imag_,     0.0,    w_real_,         0.0,
                             0.0,     w_imag_,        0.0,     w_real_)
            w.append(w_)

        return w

    def _sample_radio_material(
        self,
        si : mi.SurfaceInteraction3f,
        k_world : mi.Vector3f,
        e_fields : mi.Vector4f,
        solid_angle : mi.Float,
        sample1 : mi.Float,
        sample2 : mi.Point2f,
        specular_reflection : bool,
        diffuse_reflection : bool,
        refraction : bool,
        active : mi.Bool
        ) -> Tuple[mi.BSDFSample3f, mi.Vector4f]:
        r"""
        Evaluates the radio material and updates the electric field accordingly

        :param si: Information about the interaction of the rays with a surface
            of the scene
        :param k_world: Direction of propagation of the incident wave in the
            world frame
        :param e_fields: Electric field Jones vector as a 4D real-valued vector
        :param solid_angle: Ray tube solid angle [sr]
        :param sample1: Random float uniformly distributed in :math:`[0,1]`.
            Used to sample the interaction type.
        :param sample2: Random 2D point uniformly distributed in
            :math:`[0,1]^2`. Used to sample the direction of diffusely reflected
            waves.
        :param specular_reflection: If set to `True`, then the specularly
            reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely
            reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are
            computed
        :param active: Mask to specify active rays

        :return: Updated electric field and sampling record
        """

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world = si.n*dr.sign(dr.dot(si.n, -k_world))
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()
        si.n = normal_world

        # Set `si.wi` to the local direction of propagation of the incident wave
        si.wi = si.to_local(k_world)

        # Context.
        # Specify the components that are required
        component = 0
        if specular_reflection:
            component |= InteractionType.SPECULAR
        if diffuse_reflection:
            component |= InteractionType.DIFFUSE
        if refraction:
            component |= InteractionType.REFRACTION
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                            type_mask=0,
                            component=component)

        # Samples and evaluate the radio material
        for i, e_field in enumerate(e_fields):
            # We use:
            #  `si.dn_du` to store the real components of the S and P
            #       coefficients of the incident field
            #  `si.dn_dv` to store the imaginary components of the S and P
            #       coefficients of the incident field
            #  `si.dp_du` to store the solid angle
            # Note that the incident field is represented in the implicit world
            #   frame
            # Real components
            si.dn_du = mi.Vector3f(e_field.x, # S
                                   e_field.y, # P
                                   0.)
            # Imag components
            si.dn_dv = mi.Vector3f(e_field.z, # S
                                   e_field.w, # P
                                   0.)
            # Solid angle
            si.dp_du = mi.Vector3f(solid_angle, 0., 0.)
            # Sample and evaluate the radio material
            sample, jones_mat = si.bsdf().sample(ctx, si, sample1, sample2,
                                                 active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)
            # Update the field by applying the Jones matrix
            e_fields[i] = mi.Vector4f(jones_mat@e_field)

        return sample, e_fields
