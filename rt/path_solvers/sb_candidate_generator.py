#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generator of paths candidates using shooting and bouncing of rays"""

import mitsuba as mi
import drjit as dr
from typing import Tuple

from sionna.rt.constants import InteractionType, MIN_SEGMENT_LENGTH
from sionna.rt.utils import spawn_ray_from_sources, fibonacci_lattice,\
    spawn_ray_to
from .sample_data import SampleData
from .paths_buffer import PathsBuffer


class SBCandidateGenerator:
    r"""
    Generates path candidates using shooting-and-bouncing of rays

    This is a callable object that returns the candidates as a
    :class:`~sionna.rt.PathsBuffer` object.

    Paths flagged as valid, i.e., for which the entry in
    ``PathsBuffer.valid`` is set to `True`, are finalized: They connect a source
    to a target. Note that only paths ending with a diffuse reflection can
    be finalized through shooting-and-bouncing of rays and are therefore flagged
    as valid.

    Specular chains, which consists only of specular reflections and/or
    refractions, are only candidates. They require additional processing,
    e.g., using the image method, to either refine them to valid paths or to
    discard them.

    Similarly, paths ending by a specular suffix also require further
    processing. Neither specular chains nor paths ending by a specular suffix
    are flagged as valid.

    This generator ensures that all specular chain candidates are uniquely
    present in the returned buffer. This uniqueness is ensured through hashing
    of the paths. Note that this can causes loss of candidates due to hash
    collisions.
    """

    # Specular chains are considered identical if they share the same
    # hash. An array is used to store the number of times that a hash has
    # been observed. If the counter is > 0, then the specular chain is not
    # considered as new and not saved as a candidate. This array is indexed by
    # taking the hash modulo the size of this array. The size of this array
    # needs therefore to be large enough to ensure that the number of collisions
    # stays low and that candidates are not discarded due to collisions. The
    # following constant gives the minimum size of this array per source.
    MIN_SPEC_COUNT_SIZE = int(1e6)

    def __init__(self):

        # Sampler for generating random numbers
        self._sampler = mi.load_dict({'type': 'independent'})

    def __call__(self,
                 mi_scene : mi.Scene,
                 src_positions : mi.Point3f,
                 tgt_positions : mi.Point3f,
                 samples_per_src : int,
                 max_num_paths_per_src : int,
                 max_depth : int,
                 los : bool,
                 specular_reflection : bool,
                 diffuse_reflection : bool,
                 refraction : bool,
                 seed : int = 1) -> PathsBuffer:
        # pylint: disable=line-too-long
        r"""
        Instantiates the paths buffer and runs the candidate generator

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param samples_per_src: Number of samples to spawn per source
        :param max_num_paths_per_src: Maximum number of candidates per source
        :param max_depth: Maximum path depths
        :param los: If set to `True`, then the LoS paths are computed
        :param specular_reflection: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are computed
        :param seed: Seed for the sampler. Defaults to 1.

        :return : Candidate paths
        """

        num_sources = dr.shape(src_positions)[1]
        num_samples = samples_per_src*num_sources
        max_num_paths = max_num_paths_per_src*num_sources

        # Set the seed of the sampler
        self._sampler.seed(seed, num_samples)

        # Allocate memory for `max_num_paths` paths.
        # After the shoot-and-bounce process, if the number of paths found is
        # below `max_num_paths`, then the tensors are shrinked.
        paths = PathsBuffer(max_num_paths, max_depth)

        # Counter indicating how many paths were found for each source.
        # To ensure that the path buffer is not filled by a single or a few
        # sources, we count the number of paths traced for each source to ensure
        # that no more than `max_num_paths_per_src` are stored.
        # This is a way to ensure that the buffer is equally allocated to all
        # sources.
        paths_counter_per_source = dr.zeros(mi.UInt, num_sources)

        # Test LoS and add valid LoS paths to `paths`
        if los:
            self._los(mi_scene, src_positions, tgt_positions, paths,
                      paths_counter_per_source)

        # Run Shooting-and-bouncing of rays if required, i.e., if max_depth > 0
        if max_depth > 0:
            self._shoot_and_bounce(mi_scene, src_positions, tgt_positions,
                    paths, samples_per_src, max_num_paths_per_src, max_depth,
                    paths_counter_per_source, specular_reflection,
                    diffuse_reflection, refraction)

        return paths

    ##################################################
    # Internal methods
    ##################################################

    @dr.syntax
    def _los(self,
             mi_scene : mi.Scene,
             src_positions : mi.Point3f,
             tgt_positions : mi.Point3f,
             paths : PathsBuffer,
             paths_counter_per_source : mi.UInt):
        # pylint: disable=line-too-long
        r"""
        Tests line-of-sight (LoS) paths and add non-obstructed ones to the
        buffer

        The buffer ``paths`` is updated in-place.

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param paths: Paths buffer. Updated in-place.
        :param paths_counter_per_source: Counts the number of paths found for each source
        """

        num_src = dr.width(src_positions)
        num_tgt = dr.width(tgt_positions)

        # Sample data
        samples_data = SampleData(num_src, num_tgt, 0)

        # Target indices
        tgt_indices = dr.arange(mi.UInt, num_tgt)
        tgt_indices = dr.tile(tgt_indices, num_src)

        # Rays origins and targets
        origins = dr.repeat(src_positions, num_tgt)
        targets = dr.tile(tgt_positions, num_src)

        # Discard LoS paths when sources and targets overlap
        length = dr.norm(targets - origins)
        valid = length > MIN_SEGMENT_LENGTH

        # Test LoS
        rays = spawn_ray_to(origins, targets)
        valid &= ~mi_scene.ray_test(rays, active=valid)

        # Store the paths
        # Index where to store the current path.
        path_ind = dr.scatter_inc(paths.paths_counter, mi.UInt(0), valid)
        paths.add_paths(mi.UInt(0), path_ind, samples_data, valid, tgt_indices,
                        rays.d, -rays.d, valid)
        # Increment the sources counters
        dr.scatter_inc(paths_counter_per_source, samples_data.src_indices,
                       valid)

    def _hash_shape_ptr(self, ptr : mi.ShapePtr) -> mi.UInt64:
        r"""
        Hashes a shape pointer to an unsigned integer

        :param ptr: Pointer to a shape

        :return: Hash value
        """

        h = dr.reinterpret_array(mi.UInt32, ptr)
        h = mi.UInt64(h)
        return h

    def _cantor_pairing(self, s : mi.UInt, p : mi.UInt) -> mi.UInt:
        r"""
        Uniquely encodes two natural numbers into a single natural number using
        the Cantor pairing function

        :param s: First integer
        :param p: Second integer

        :return: Pairing function output
        """

        h = s*s + 3*s + 2*s*p + p + p*p
        h //= 2
        return h

    def _hash_intersection(self,
                           shape_ptr : mi.ShapePtr,
                           prim_ind : mi.UInt,
                           int_type : mi.UInt) -> mi.UInt:
        r"""
        Hashes a pointer to a shape, a primitive index, and an interaction type

        :param shape_ptr: Pointer to a shape
        :param prim_ind: Index of primitive
        :param int_type: Intersection type

        :return: Hash value
        """

        shape_hash = self._hash_shape_ptr(shape_ptr)
        a = self._cantor_pairing(shape_hash + 1, prim_ind + 1)
        inter_hash = self._cantor_pairing(a, int_type)
        return inter_hash


    def _polynomial_hashing(self,
                            interaction_hash : mi.UInt,
                            path_hash : mi.UInt) -> mi.UInt:
        r"""
        Updates the hash of a path ``path_hash`` with the interaction hash
        ``interaction_hash`` using polynomial hashing, i.e.,

        .. code-block:: Python

            hash[n] = hash[n-1]*p + i

        where ``hash[n-1]`` is the provided path hash (``path_hash``),
        ``hash[n]`` the returned updated path hash, ``i`` the interaction hash
        (``interaction_hash``), and ``p`` a prime number

        :param interaction_hash: Interaction hash
        :param path_hash: Current path hash

        :return: Updated path hash
        """

        prime = mi.UInt64(1373)
        path_hash = path_hash*prime + interaction_hash

        return path_hash

    def _shoot_and_bounce(self,
                          mi_scene : mi.Scene,
                          src_positions : mi.Point3f,
                          tgt_positions : mi.Point3f,
                          paths : PathsBuffer,
                          samples_per_src : int,
                          max_num_paths_per_src : int,
                          max_depth : int,
                          paths_counter_per_source : mi.UInt,
                          specular_reflection : bool,
                          diffuse_reflection : bool,
                          refraction : bool):
        # pylint: disable=line-too-long
        r"""
        Executes shooting-and-bouncing of rays

        The paths buffer ``path`` is updated in-place.

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param paths: Buffer storing the candidate paths. Updated in-place.
        :param samples_per_src: Number of samples spawn per source
        :param max_num_paths_per_src: Maximum number of candidates per source
        :param max_depth:  Maximum path depths
        :param paths_counter_per_source: Counts the number of paths found for each source
        :param specular_reflection: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are computed
        """

        num_sources = dr.shape(src_positions)[1]

        # Counter indicating how many occurrences of a specular chain was found.
        # Specular chains are considered identical if they share the same
        # hash. Taking the hash modulo the size of the following array is used
        # to index this array and increment the counter. If the counter is > 0,
        # then the specular chain is not considered as new and not stored.
        # The size of the following array needs therefore to be large enough
        # to ensure that the number of collisions stays low and that candidates
        # are not discarded due to collisions.
        # The size of the array is set to:
        #  max(max_num_paths, MIN_SPEC_COUNTER_SIZE*num_sources)
        spec_counter_size = dr.maximum(max_num_paths_per_src,
                                       SBCandidateGenerator.MIN_SPEC_COUNT_SIZE)
        specular_chain_counter = dr.zeros(mi.UInt,
                                          spec_counter_size*num_sources)

        # Runs the shooting-and-bouncing of rays loop
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            self._shoot_and_bounce_loop(mi_scene, src_positions, tgt_positions,
                samples_per_src, max_num_paths_per_src, max_depth, paths,
                paths_counter_per_source, specular_chain_counter,
                specular_reflection, diffuse_reflection, refraction,
                self._sampler)

    @dr.syntax
    def _shoot_and_bounce_loop(self,
                               mi_scene : mi.Scene,
                               src_positions : mi.Point3f,
                               tgt_positions : mi.Point3f,
                               samples_per_src : int,
                               max_num_paths_per_src : int,
                               max_depth : int,
                               paths : PathsBuffer,
                               paths_counter_per_source : mi.UInt,
                               specular_chain_counter : mi.UInt,
                               specular_reflection : bool,
                               diffuse_reflection : bool,
                               refraction : bool,
                               sampler : mi.Sampler):
        # pylint: disable=line-too-long
        r"""
        Executes shooting-and-bouncing of rays

        The paths buffer ``path`` is updated in-place.

        :param mi_scene: Mitsuba scene
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param samples_per_src: Number of samples spawn per source
        :param max_num_paths_per_src: Maximum number of candidates per source
        :param max_depth:  Maximum path depths
        :param paths: Buffer storing the candidate paths. Updated in-place.
        :param paths_counter_per_source: Counts the number of paths found for each source
        :param specular_reflection: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are computed
        :param sampler: Sampler used to generate pseudo-random numbers
        """

        num_sources = dr.shape(src_positions)[1]
        num_targets = dr.shape(tgt_positions)[1]
        num_samples = samples_per_src*num_sources
        spec_counter_size = dr.shape(specular_chain_counter)[0]//num_sources

        # Structure storing the sample data, which is used to build the paths
        samples_data = SampleData(num_sources, samples_per_src, max_depth)

        # Rays
        ray = spawn_ray_from_sources(fibonacci_lattice, samples_per_src,
                                     src_positions)

        # Store direction of departure
        k_tx = dr.copy(ray.d)

        # Boolean indicating if the sample is a specular chain, i.e., if it
        # consists only of specular chains.
        specular_chain = dr.full(mi.Bool, True, num_samples)

        # Hash of the paths.
        # It is computed only for specular chains, and used to not duplicate
        # specular chain candidates.
        # 64bit integer is used for hashing.
        path_hash = dr.zeros(mi.UInt64, num_samples)

        # Current depth
        depth = dr.full(mi.UInt, 1, num_samples)

        # Mask indicating which rays are active
        active = dr.full(mi.Mask, True, num_samples)

        # Note: here and in the inner loop, we explicitly exclude some non-state
        # variables from the loop state so that DrJit doesn't have to trace
        # the loop body twice to figure it out.
        while dr.hint(active, label="shoot_and_bounce", exclude=[
            specular_chain_counter,
            paths_counter_per_source,
        ]):

            ########################################################
            # Test intersection with the scene and evaluate the
            # intersection
            ########################################################

            # Test intersection with the scene
            si_scene = mi_scene.ray_intersect(ray, coherent=True,
                                              ray_flags=mi.RayFlags.Minimal,
                                              active=active)

            # Deactivate rays that didn't hit the scene, i.e., that bounce-out
            # of the scene
            active &= si_scene.is_valid()

            # Samples the radio material
            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()
            s, n = self._sample_radio_material(si_scene, ray.d, sample1,
                    sample2, specular_reflection, diffuse_reflection,
                    refraction, active)
            # Direction of propagation of scattered wave in implicit world
            # frame
            k_world = s.wo
            # Interaction type
            int_type = dr.select(active, s.sampled_component,
                                 InteractionType.NONE)
            # Disable paths if a NONE interaction was sampled.
            # This happens if no interaction type is enabled
            active &= (int_type != InteractionType.NONE)

            # Is this interaction a specular reflection?
            specular = int_type == InteractionType.SPECULAR

            # Is this interaction a transmission
            transmission = int_type == InteractionType.REFRACTION

            # Is this interaction a diffuse reflection
            diffuse = int_type == InteractionType.DIFFUSE

            # Is the sample a specular chain?
            # A specular chain consists only of specular reflections or
            # transmission
            specular_chain &= active & (specular | transmission)

            ########################################################
            # Update samples data
            ########################################################

            # Update the samples data
            samples_data.insert(depth, int_type, si_scene.shape,
                                si_scene.prim_index, si_scene.p)

            ########################################################
            # Store the paths.
            # A path is stored if:
            # - It is a new specular chain
            # - It is valid, i.e., it connects to a target
            ########################################################

            # If this path is a specular chain, then we hash it to ensure it
            # is a new paths.
            # Hash the interaction
            inter_hash = self._hash_intersection(si_scene.shape,
                                                 si_scene.prim_index,
                                                 int_type)
            # Hash of the path
            path_hash = self._polynomial_hashing(inter_hash, path_hash)

            # Loop over all targets.

            # Target index
            t = mi.UInt(0)
            while dr.hint(t < num_targets, label="shoot_and_bounce_inner",
                          exclude=[specular_chain_counter,
                                   paths_counter_per_source]):
                # Position of the target with index t
                tgt_position = dr.gather(mi.Point3f, tgt_positions, t)

                # Test line-of-sight with the target from the current
                # interaction point
                los_ray = si_scene.spawn_ray_to(tgt_position)
                los_blocked = mi_scene.ray_test(los_ray, active=active)
                los_visible = ~los_blocked

                # If the interaction is valid and if the target is visible from
                # the intersection point, then the path is marked as valid.

                # It is also required that the target is on the same side of
                # the intersected surface than the incident wave.
                # `n`` is the normal to the intersected surface oriented towards
                # the incident half-space
                # `los_ray.d` is from the intersection point to the target
                target_incident_side = dr.dot(n, los_ray.d) > 0.
                valid = los_visible & diffuse & target_incident_side

                # If this path is a specular chain, then we hash it to ensure
                # that it is a new paths.
                # A specular chain is only considered as candidate if the
                # intersection point is in LoS with the target. This condition
                # is used as an heuristic to reduce the number of candidates.
                # It also helps to reduce the number of access to the hash table
                # storing the specular chain counter, and therefore reduces th
                # number of collisions.
                new_specular = specular_chain & los_visible

                # A specular chain is considered as new, and therefore should
                # be stored in the `path` structure, if its hash has not been
                # already observed. To ensure that the path has not
                # already been stored for the target `t`, we combine it with
                # the path hash
                # If the sample is a specular chain, then the counter
                # corresponding to its hash is increased, and we check that the
                # counter value previous to its increment equals 0.
                path_target_hash = self._cantor_pairing(path_hash, t)
                counter_ind = path_target_hash % spec_counter_size
                counter_ind += spec_counter_size*samples_data.src_indices
                samples_counter = dr.scatter_inc(specular_chain_counter,
                                                 counter_ind, new_specular)
                new_specular &= samples_counter == 0

                # Store the paths

                store = active & (valid | new_specular)

                # Increment the per source path counter
                num_path_per_src = dr.scatter_inc(paths_counter_per_source,
                                                  samples_data.src_indices,
                                                  store)
                # If we exceeded the specified maximum number of paths,
                # then paths are discarded
                store &= num_path_per_src < max_num_paths_per_src

                # Index where to store the current path.
                path_ind = dr.scatter_inc(paths.paths_counter, mi.UInt(0),
                                          store)

                # Store the paths
                paths.add_paths(depth, path_ind, samples_data, valid, t, k_tx,
                                -los_ray.d, store)

                t += 1

            ####################################
            # Prepare next iteration
            ####################################

            # Deactivate rays if the maximum depth is reached and
            # the ones set as valid
            depth += 1
            active &= (depth <= max_depth)

            # Spawn rays for next iteration
            ray = si_scene.spawn_ray(d=k_world)

            # Reset the value of specular_chain in case of a diffuse reflection
            specular_chain |= diffuse

    def _sample_radio_material(self,
                               si : mi.SurfaceInteraction3f,
                               k_world : mi.Vector3f,
                               sample1 : mi.Float,
                               sample2 : mi.Point2f,
                               specular_reflection : bool,
                               diffuse_reflection : bool,
                               refraction : bool,
                               active : mi.Bool
                               ) -> Tuple[mi.BSDFSample3f, mi.Normal3f]:
        # pylint: disable=line-too-long
        r"""
        Samples the radio material of the intersected object

        This function prepares the inputs of the ``sample()`` method of the
        radio material, calls it, and then returns its output.

        :param si: Information about the interaction
        :param k_world: Direction of propagation of the incident wave in the world frame
        :param sample1: Random float uniformly distributed in :math:`[0,1]`. Used to sample the interaction type.
        :param sample2: Random 2D point uniformly distributed in :math:`[0,1]^2`. Used to sample the direction of diffusely reflected waves.
        :param specular_reflection: If set to `True`, then the specularly reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are computed
        :param active: Mask to specify active rays

        :return: Sampling record
        :return: Normal to the incident surface in the world frame
        """

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world = si.n*dr.sign(dr.dot(si.n, -k_world))
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()
        si.n = normal_world

        # Set `si.wi` to the direction of propagation of the incident wave in
        # the local frame
        si.wi = si.to_local(k_world)

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

        # Samples the radio material
        sample, _ = si.bsdf().sample(ctx, si, sample1, sample2, active)

        return sample, normal_world
