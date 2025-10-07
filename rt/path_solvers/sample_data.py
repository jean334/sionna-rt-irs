#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Stores shooting and bouncing of rays samples data"""

import mitsuba as mi
import drjit as dr
from dataclasses import dataclass
from typing import Tuple


class SampleData:
    r"""
    Class used to store shoot-and-bounce samples data

    A sample is a path spawn from a source when running shooting and
    bouncing of rays. However, to distinguish these paths from the ones stored
    in a :class:`~sionna.rt.PathsBuffer`, which are further processed and/or
    returned by the solver, the paths spawn during shooting and bouncing of rays
    are referred to as "samples".

    Each sample can lead to the recoring in a :class:`~sionna.rt.PathsBuffer` of
    one or more paths.

    :param num_sources: Number of sources
    :param samples_per_src: Number of samples spawn per source
    :param max_depth: Maximum depth
    """

    @dataclass
    class SampleDataFields:
        r"""
        Dataclass used to store information of a single sample and for a single
        interaction

        :data interaction_type: Type of interaction
        :data shape: Pointer to the intersected shape as an unsigned integer
        :data primitive: Index of the intersected primitive
        :data vertex: Coordinates of the intersection point with the scene
        """

        interaction_type    : mi.UInt
        shape               : mi.UInt
        primitive           : mi.UInt
        vertex              : mi.Point3f

    def __init__(self,
                 num_sources : int,
                 samples_per_src : int,
                 max_depth : int):

        # Size of the array
        # If `max_depth` is 0, we need to allocate at least one element to
        # store the LoS path data
        array_size = max(1, max_depth)

        # Index of the source corresponding to samples.
        # Samples-first ordering is used, i.e., the samples are ordered as
        # follows:
        # [source_0_samples..., source_1_samples..., ...]
        src_indices = dr.arange(mi.UInt, 0, num_sources)
        src_indices = dr.repeat(src_indices, samples_per_src)
        self._src_indices = src_indices

        # Structure storing the data for a single sample.
        # A DrJit local memory is used to enable read-after-write dependencies.
        # That implies that a buffer is created for each thread, which ray trace
        # a single sample, to store information about this sample.
        # The size of the buffer is set to `max_depth`, as a sample can consists
        # of up to that many interactions with the scene.
        # See https://drjit.readthedocs.io/en/latest/misc.html#local-memory
        self._local_mem = dr.alloc_local(SampleData.SampleDataFields,
                                         array_size)

    def insert(self,
               depth : mi.UInt,
               interaction_types : mi.UInt,
               shapes : mi.ShapePtr,
               primitives : mi.UInt,
               vertices : mi.Point3f):
        # pylint: disable=line-too-long
        r"""
        Stores interaction data for depth ``depth``

        :param depth: Depth for which to store sample data
        :param interaction_types: Type of interaction represented using :class:`~sionna.rt.constants.InteractionType`
        :param shapes: Pointers to the intersected shapes
        :param primitives: Indices of the intersected primitives
        :param vertices: Coordinates of the intersection points
        """
        # Depth ranges from 1 to max_depth
        index = depth - 1

        # Shape pointers are converted to unsigned integers
        shapes = dr.reinterpret_array(mi.UInt, shapes)

        # Store data in the buffer
        data = SampleData.SampleDataFields(interaction_types, shapes,
                                           primitives, vertices)
        self._local_mem[index] = data

    def get(self,
            depth : mi.UInt) -> Tuple[mi.UInt, mi.UInt, mi.UInt, mi.Point3f]:
        # pylint: disable=line-too-long
        r"""
        Returns data about the sample for depth ``depth``

        :param depth: Depths for which to return sample data

        :return: Type of interaction represented using :class:`~sionna.rt.constants.InteractionType`
        :return: Pointers to the intersected shapes
        :return: Indices of the intersected primitives
        :return: Coordinates of the intersection points
        """
        index = depth - 1
        data = self._local_mem[index]

        interaction_types = data.interaction_type
        shapes = data.shape
        primitives = data.primitive
        vertices = data.vertex

        return interaction_types, shapes, primitives, vertices

    @property
    def src_indices(self):
        r"""Indices of the sources from which samples originate

        :type: :py:class:`mi.UInt`
        """
        return self._src_indices
