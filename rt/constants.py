#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Constants of Sionna RT"""

import drjit as dr
import mitsuba as mi


class InteractionType:
    """
    Constants representing the type of interaction
    """

    #: No interaction
    NONE            = 0

    #: Specular reflection
    SPECULAR        = 1 << 0 # 1

    #: Diffuse reflection
    DIFFUSE         = 1 << 1 # 2

    #: Refraction
    REFRACTION    = 1 << 2 # 4

#: Default frequency [Hz]
DEFAULT_FREQUENCY = 3.5e9

#: Default bandwidth [Hz]
DEFAULT_BANDWIDTH = 1e6 # Hz

#: Default temperature [K]
DEFAULT_TEMPERATURE = 293

#: Default transmit power [dBm]
DEFAULT_TRANSMIT_POWER_DBM = 44

#: Default background color for the preview
DEFAULT_PREVIEW_BACKGROUND_COLOR = "white"

#: Default color used to visualize transmitters in preview and rendering (R,G,B)
DEFAULT_TRANSMITTER_COLOR = (1.0, 0.0, 0.0)
#: Default color used to visualize receivers in preview and rendering (R,G,B)
DEFAULT_RECEIVER_COLOR = (0.4, 0.8, 0.4)

#: Color used to visualize line-of-sight path segments (R,G,B)
LOS_COLOR = (0.5, 0.5, 0.5)
#: Color used to visualize specularly reflected path segments (R,G,B)
SPECULAR_COLOR = (0.6, 0.6, 1.0)
#: Color used to visualize diffusely reflected path segments (R,G,B)
DIFFUSE_COLOR = (0.6, 1., 0.6)
#: Color used to visualize transmitted path segments (R,G,B)
REFRACTION_COLOR = (1., 0.6, 0.6)

# Maps interaction type to color
INTERACTION_TYPE_TO_COLOR = {
    None: LOS_COLOR,  # E.g. line-of-sight
    InteractionType.SPECULAR: SPECULAR_COLOR,
    InteractionType.DIFFUSE: DIFFUSE_COLOR,
    InteractionType.REFRACTION: REFRACTION_COLOR,
}

#: Constant representing the default thickness of radio materials [m]
DEFAULT_THICKNESS = 0.1

#: Constant representing an invalid shape
INVALID_SHAPE           = 0xFFFFFFFF
#: Constant representing an invalid primitive
INVALID_PRIMITIVE       = 0xFFFFFFFF

#: Numerical stability constant
EPSILON_FLOAT = dr.epsilon(mi.Float) * 100.

# pylint: disable=line-too-long
#: Minimum length of a path segment [m]. Paths with a segment shorter than this threshold are discarded.
MIN_SEGMENT_LENGTH = 1e-2
