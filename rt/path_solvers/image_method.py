#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of the image method"""

import drjit as dr
import mitsuba as mi
from typing import Tuple

from sionna.rt.utils import spawn_ray_to, offset_p
from sionna.rt.constants import InteractionType, MIN_SEGMENT_LENGTH
from .paths_buffer import PathsBuffer


class ImageMethod:
    r"""
    Image method for evaluating specular chains and specular suffixes candidates

    A specular chain (suffix) is a path (suffix) that consists only of specular
    reflections and refractions.

    This class processes the candidate specular suffixes and specular chains
    using the image method to compute valid paths from the candidates specular
    chains and suffixes. It consists in 5 steps:

    1. The depth at which the specular suffix starts is determined.
    This depth corresponds to the lowest depth from which the path consists only
    of specular reflections and refraction. For specular chains, this depth
    equals to one.

    2. The source of the specular suffix is determined. For specular chains, it
    is simply the source of the path. For specular suffixes, it is
    the intersection point preceding the specular suffix.

    4. Image computations: Images of the sources are computed by reflecting them
    on surfaces on which a specular reflection occurred. Refraction events
    are ignored during this step.

    5. Backtracking: Tracing is done backward from the targets to the sources,
    by spawning rays toward the images, first from the target, then from the
    intersection points of the rays with the scene. These intersection
    points are the final vertices of the paths for specular reflections and
    refractions. If intersections with other primitives than the ones computed
    during the candidates generation are found, then the paths are discarded,
    i.e., flagged as a non-valid candidate.
    """

    def __init__(self):

        # Dr.Jit mode for running the loops that implement the image method
        # Symbolic mode is the fastest mode but does not currently support
        # backpropagation of gradients
        self._loop_mode = "symbolic"

    @property
    def loop_mode(self):
        r"""Get/set the Dr.Jit mode used to evaluate the loops that implement
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.

        :type: "evaluated" | "symbolic"
        """
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._loop_mode = mode

    def __call__(self,
                 scene : mi.Scene,
                 paths : PathsBuffer,
                 src_positions : mi.Point3f,
                 tgt_positions : mi.Point3f) -> PathsBuffer:
        r"""
        Exectues the image method

        :param scene: Mitsuba scene
        :param paths: Candidate paths
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets

        :return: Processed paths
        """

        # Stop immediately if there are no paths
        if paths.buffer_size == 0:
            return paths

        # Candidates are the paths not marked as valid during the candidates
        # generation process
        valid_candidate = ~paths.valid

        # Gather the source and target of every path
        paths_sources, paths_targets\
            = self._gather_paths_sources_targets(paths, src_positions,
                                                 tgt_positions, valid_candidate)

        # Depth at which the specular suffixes start
        sf_start_depth = self._specular_chain_start_depth(paths,
                                                          valid_candidate)

        # Position of the specular suffixes sources
        sf_source = self._get_sf_source(paths, paths_sources, sf_start_depth,
                                        valid_candidate)

        ## Image method ##

        # Computes the images
        self._compute_images(paths, sf_source, sf_start_depth,  valid_candidate)

        # Backtrack
        valid_candidate = self._backtrack(scene, paths, paths_targets,
                                          sf_source, sf_start_depth,
                                          valid_candidate)

        # ## ## ## ## ## ##

        # Update the candidate valid status
        paths.valid |= valid_candidate

        return paths

    ##################################################
    # Internal methods
    ##################################################

    def _gather_paths_sources_targets(self,
                                      paths,
                                      src_positions : mi.Point3f,
                                      tgt_positions : mi.Point3f,
                                      valid_candidate : mi.Bool
                                      ) -> Tuple[mi.Point3f, mi.Point3f]:
        r"""
        Gathers the sources and targets of every paths

        :param paths: Candidate paths
        :param src_positions: Positions of the sources
        :param tgt_positions  Positions of the targets
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Sources and targets of all paths
        """

        paths_sources = dr.gather(mi.Point3f, src_positions,
                                  paths.source_indices, valid_candidate)
        paths_targets = dr.gather(mi.Point3f, tgt_positions,
                                  paths.target_indices, valid_candidate)
        return paths_sources, paths_targets

    @dr.syntax
    def _specular_chain_start_depth(self,
                                    paths : PathsBuffer,
                                    valid_candidate : mi.Bool) -> mi.UInt:
        r"""
        Determines the depth at which the specular suffixes start

        For specular chains, the returned depth is 0.

        :param paths: Candidate paths
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Depths at which the specular suffixes start
        """

        max_depth = paths.max_depth
        max_num_paths = paths.buffer_size

        # Array for storing the depth at which the specular suffix starts
        sf_start_depth = dr.full(mi.UInt, max_depth, max_num_paths)

        active = dr.copy(valid_candidate)
        depth = mi.UInt(max_depth)
        while dr.hint(active, mode=self.loop_mode):

            int_type = paths.get_interaction_type(depth, active)
            specular = int_type == InteractionType.SPECULAR
            refraction = int_type == InteractionType.REFRACTION
            none = int_type == InteractionType.NONE

            # If not in the specular suffix anymore, then deactivate the path to
            # stop updating the corresponding entry in `sf_start_depth`
            active &= (specular | refraction | none)

            # Update the starting point if the ray is active
            sf_start_depth = dr.select(active, depth, sf_start_depth)

            depth -= 1
            active &= depth > 0

        return sf_start_depth

    def _get_sf_source(self,
                       paths : PathsBuffer,
                       paths_sources  : mi.Point3f,
                       sf_start_depth : mi.UInt,
                       valid_candidate : mi.Bool) -> mi.Point3f:
        r"""
        Returns the sources positions of the specular suffixes, which is
        * The source of the path if the path is a specular chain
        * The vertex of the last diffuse reflection if the path is not a
            specular chain but ends with a specular suffix

        :param paths: Candidate paths
        :param paths_sources: Sources of every paths
        :param sf_start_depth: Depths at which the specular suffixes start
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Positions of the sources of the specular suffixes
        """

        sf_source = sf_start_depth == 1

        last_diff_depth = sf_start_depth - 1
        last_diff_depth = dr.select(sf_source, 1, last_diff_depth)

        vertex = paths.get_vertex(last_diff_depth, valid_candidate)
        sf_source = dr.select(sf_source, paths_sources, vertex)

        return sf_source

    @dr.syntax
    def _compute_images(self,
                        paths : PathsBuffer,
                        sf_source : mi.Point3f,
                        sf_start_depth : mi.UInt,
                        valid_candidate : mi.Bool):
        r"""
        Computes the images of the sources

        This is the first step of the image method.
        The images coordinates are stored in ``path.vertices``.

        :param paths: Candidate paths
        :param sf_source: Positions of the sources of the specular suffixes
        :param sf_start_depth: Depths at which the specular suffixes starts
        :param valid_candidate: Flag specifying candidates marked as valid
        """

        max_depth = paths.max_depth
        num_paths = paths.buffer_size

        active = dr.copy(valid_candidate)
        depth = dr.full(mi.UInt, 1, num_paths)
        image = dr.copy(sf_source)
        normal = dr.zeros(mi.Vector3f, num_paths)
        while dr.hint(active, mode=self.loop_mode):

            int_type = paths.get_interaction_type(depth, active)
            specular = (int_type == InteractionType.SPECULAR) & active
            none = (int_type == InteractionType.NONE) & active
            # Deactivate the ray if no interaction
            active &= ~none

            # Flag indicating if this specular interaction is part of the
            # specular suffix
            in_sf = specular & (depth >= sf_start_depth)

            # Next vertex
            vertex = paths.get_vertex(depth, active)
            # Current normal
            normal = paths.get_normal(depth, active)

            # Compute the image of the current vertex with respect to the
            # intersected primitive
            x = dr.dot(image - vertex, normal)
            image = dr.select(in_sf, image - 2.*x*normal, image)

            # Use path vertices as a buffer to store the images
            paths.set_vertex(depth, image, in_sf)

            depth += 1
            active &= depth <= max_depth

    @dr.syntax
    def _backtrack(self,
                   mi_scene : mi.Scene,
                   paths : PathsBuffer,
                   paths_targets : mi.Point3f,
                   sf_source : mi.Point3f,
                   sf_start_depth : mi.UInt,
                   valid_candidate : mi.Bool) -> mi.Bool:
        r"""
        Traces backwards from the targets to the images of the sources

        This function assumes that ``_compute_images()`` has already been
        executed.

        When backtracking, this function does the following:
        * It computes the path vertices, i.e., the final intersection point
        of valid candidates with the scene
        * It updates the ``valid_candidate`` array by flagging as invalid
        candidates that are occluded

        :param mi_scene: Mitsuba scene
        :param paths: Candidate paths
        :param paths_targets: Positions of the targets
        :param sf_source: Positions of the sources of the specular suffixes
        :param sf_start_depth: Depths at which the specular chain starts
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Updated ``valid_candidate`` array
        """

        max_depth = paths.max_depth

        @dr.syntax
        def find_next_spec_depth(self, paths, current_depth, current_active,
                                 min_depth):
            depth = current_depth + 1
            active = dr.copy(current_active)
            while dr.hint(active, mode=self.loop_mode):
                depth -= 1
                active &= (depth >= min_depth)

                int_type = paths.get_interaction_type(depth, active)
                specular = int_type == InteractionType.SPECULAR
                active &= (~specular)

            return depth

        active = dr.copy(valid_candidate)
        depth = dr.full(mi.UInt, max_depth, paths.buffer_size)
        was_none = dr.full(mi.Bool, True, paths.buffer_size)
        vertex = dr.copy(paths_targets)
        normal = dr.zeros(mi.Normal3f, paths.buffer_size)
        while dr.hint(active, mode=self.loop_mode):

            # Depth of the next specular reflection beforehand (as we backtrack)
            next_spec_depth = find_next_spec_depth(self, paths, depth, active,
                                                   sf_start_depth)

            # The intersection is initially valid if the ray is active
            # and there is an intersection
            int_type = paths.get_interaction_type(depth, active)
            none = int_type == InteractionType.NONE
            valid_inter = active & ~none

            # Read the next image.
            # If there is no specular reflection beforehand in the specular
            # suffix, then use the source as the next image
            spec_next = next_spec_depth >= sf_start_depth
            image = paths.get_vertex(next_spec_depth, valid_inter & spec_next)
            image = dr.select(spec_next, image, sf_source)

            # To avoid ray leakage in case where the path vertex
            # (si_scene.p in the intersection test below) is located exactly at
            # the intersection between two perpendicular shapes, a small offset
            # is added to the image along the corresponding normal
            normal_image = paths.get_normal(depth, valid_inter & spec_next)
            image = offset_p(image, vertex - image, normal_image)

            # Spawn a ray from the vertex towards the next image
            ray = spawn_ray_to(vertex, image, normal)
            si_scene = mi_scene.ray_intersect(ray,
                                              ray_flags=mi.RayFlags.Minimal,
                                              coherent=True,
                                              active=valid_inter)

            # Check the that the intersection is valid. It is if:
            # - There is an intersection, and
            # - The intersected primitive is the one detected during candidate
            # generation
            # - The segment lengt is above a pre-defined threshold
            valid_inter &= si_scene.is_valid()
            expected_shape = paths.get_shape(depth, valid_inter)
            si_shape = dr.reinterpret_array(mi.UInt, si_scene.shape)
            expected_prim_ind = paths.get_primitive(depth, valid_inter)
            valid_inter &= (expected_shape == si_shape)
            valid_inter &= (expected_prim_ind == si_scene.prim_index)
            valid_inter &= (si_scene.t > MIN_SEGMENT_LENGTH)

            # If the intersection if valid, then stores the intersection point
            # as the path vertex, and update the direction of arrival
            paths.set_vertex(depth, si_scene.p, valid_inter)
            paths.set_angles_rx(ray.d, valid_inter & was_none)
            # If the intersection is not valid, discard the candidate
            # If there was no intersection (none == True), then we did not
            # enter yet the specular suffix
            valid_candidate &= valid_inter | none

            depth -= 1
            active &= (depth  >= sf_start_depth) & valid_candidate
            vertex = dr.select(valid_inter, si_scene.p, vertex)
            normal = dr.select(valid_inter, si_scene.n, normal)
            was_none = dr.copy(none)

        # Test visibility from the last vertex to the source of the specular
        # suffix
        ray = spawn_ray_to(vertex, sf_source, normal)
        valid_candidate &= ~mi_scene.ray_test(ray, active=valid_candidate)
        # If the candidate is valid, then update the direction of depature
        paths.set_angles_tx(-ray.d, valid_candidate)

        return valid_candidate
