#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""EM utilities of Sionna RT"""

import drjit as dr
import mitsuba as mi
from typing import Tuple
from scipy.constants import epsilon_0

from .misc import complex_sqrt


def complex_relative_permittivity(
    eta_r : mi.Float,
    sigma : mi.Float,
    omega : mi.Float
) -> mi.Complex2f:
    r"""
    Computes the complex relative permittivity of a material
    as defined in :eq:`eta`

    :param eta_r: Real component of the relative permittivity
    :param sigma: Conductivity [S/m]
    :param omega: Angular frequency [rad/s]
    """

    eta_i = sigma*dr.rcp(omega*epsilon_0)
    eta = mi.Complex2f(eta_r, -eta_i)
    return eta

def fresnel_reflection_coefficients_simplified(
    cos_theta : mi.Float,
    eta : mi.Complex2f
) -> Tuple[mi.Complex2f, mi.Complex2f]:
    # pylint: disable=line-too-long
    r"""
    Computes the Fresnel transverse electric and magnetic reflection
    coefficients assuming an incident wave propagating in vacuum :eq:`fresnel_vac`

    :param cos_theta: Cosine of the angle of incidence
    :param eta:  Complex-valued relative permittivity of the medium upon which the wave is incident

    :return: Transverse electric :math:`r_{\perp}` and magnetic :math:`r_{\parallel}` Fresnel reflection coefficients
    """

    sin_theta_sqr = 1. - cos_theta*cos_theta # sin^2(theta)
    a = complex_sqrt(eta - sin_theta_sqr)

    # TE coefficient
    r_te = (cos_theta - a)*dr.rcp(cos_theta + a)

    # TM coefficient
    r_tm = (eta*cos_theta - a)*dr.rcp(eta*cos_theta + a)

    return r_te, r_tm

def itu_coefficients_single_layer_slab(
    cos_theta : mi.Float,
    eta : mi.Complex2f,
    d : mi.Float,
    wavelength : mi.Float
    ) -> Tuple[mi.Complex2f, mi.Complex2f, mi.Complex2f, mi.Complex2f]:
    # pylint: disable=line-too-long
    r"""
    Computes the single-layer slab Fresnel transverse electric and
    magnetic reflection and refraction coefficients assuming the incident wave
    propagates in vacuum using recommendation ITU-R P.2040 [ITU_R_2040_3_]

    More precisely, this function implements equations (43) and (44) from
    [ITU_R_2040_3]_.

    :param cos_theta: Cosine of the angle of incidence
    :param eta: Complex-valued relative permittivity of the medium upon which the wave is incident
    :param d: Thickness of the slab [m]
    :param wavelength:  Wavelength [m]

    :return: Transverse electric reflection coefficient :math:`R_{eTE}`, transverse magnetic reflection coefficient :math:`R_{eTM}`, transverse electric refraction coefficient :math:`T_{eTE}`, and transverse magnetic refraction coefficient :math:`T_{eTM}`
    """

    # sin^2(theta)
    sin_theta_sqr = 1. - dr.square(cos_theta)

    # Compute `q` - Equation (44)
    q = dr.two_pi*d*dr.rcp(wavelength)*complex_sqrt(eta - sin_theta_sqr)

    # Simplified Fresnel coefficients - Equations (37a) and (37b)
    r_te_p, r_tm_p = fresnel_reflection_coefficients_simplified(cos_theta, eta)
    # Squared Fresnel coefficients
    r_te_p_sqr = dr.square(r_te_p)
    r_tm_p_sqr = dr.square(r_tm_p)

    # exp(-jq) and exp(-j2q)
    exp_j_q = dr.exp(mi.Complex2f(0., -1.)*q)
    exp_j_2q = dr.exp(mi.Complex2f(0., -2.)*q)

    # Denominator of Fresnel coefficient
    denom_te = 1. - r_te_p_sqr*exp_j_2q
    inv_denom_te = dr.rcp(denom_te)
    denom_tm = 1. - r_tm_p_sqr*exp_j_2q
    inv_denom_tm = dr.rcp(denom_tm)

    # Reflection coefficients - Equation (43a)
    r_te = r_te_p*(1. - exp_j_2q)*inv_denom_te
    r_tm = r_tm_p*(1. - exp_j_2q)*inv_denom_tm
    # Transmission coefficient - Equation (43b)
    t_te = (1. - r_te_p_sqr)*exp_j_q*inv_denom_te
    t_tm = (1. - r_tm_p_sqr)*exp_j_q*inv_denom_tm

    return r_te, r_tm, t_te, t_tm
