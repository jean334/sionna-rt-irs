# SPDX-FileCopyrightText: Copyright (c) 2025 Jean ACKER.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import mitsuba as mi
import drjit as dr
from typing_extensions import Tuple

def _scalar_p3(p: mi.Point3f | mi.ScalarPoint3f) -> mi.ScalarPoint3f:
    if isinstance(p, mi.ScalarPoint3f):
        return p
    # extract lane 0 as scalar components (no numpy / float casts)
    return mi.ScalarPoint3f(dr.slice(p.x, 0), dr.slice(p.y, 0), dr.slice(p.z, 0))

def _scalar_p2(p: mi.Point2f | mi.ScalarPoint2f) -> mi.ScalarPoint2f:
    if isinstance(p, mi.ScalarPoint2f):
        return p
    return mi.ScalarPoint2f(dr.slice(p.x, 0), dr.slice(p.y, 0))

def mitsuba_rectangle_to_world(center: mi.Point3f,
                               orientation: mi.Point3f,  # radians, order Z→Y→X
                               size: mi.Point2f,         # (width, height) [m]
                               ris: bool = False) -> mi.ScalarTransform4f:
    """
    Build a scalar `to_world` for a Mitsuba rectangle from possibly vectorized inputs.
    Rotation order: Z (yaw), then Y (pitch), then X (roll).
    If `ris=True`, rotate default rectangle normal (+Z) to +X and swap width/height.
    """
    c   = _scalar_p3(center)
    ang = _scalar_p3(orientation)
    sz  = _scalar_p2(size)

    # radians → degrees (Mitsuba rotate expects degrees)
    rad2deg = 180.0 / dr.pi
    yaw   = ang.x * rad2deg   # around Z
    pitch = ang.y * rad2deg   # around Y
    roll  = ang.z * rad2deg   # around X

    T = mi.ScalarTransform4f  # alias

    # Start from identity and apply instance methods
    trans = T()  # identity
    trans = trans.translate(c)
    trans = trans @ T().rotate(mi.ScalarVector3f(0, 0, 1), yaw)
    trans = trans @ T().rotate(mi.ScalarVector3f(0, 1, 0), pitch)
    trans = trans @ T().rotate(mi.ScalarVector3f(1, 0, 0), roll)

    if ris:
        # rotate +Z (rect normal) to +X
        trans = trans @ T().rotate(mi.ScalarVector3f(0, 1, 0), 90.0)
        # swap width/height to match axes after the RIS rotation
        sz = mi.ScalarPoint2f(sz.y, sz.x)

    # Default Mitsuba rectangle spans [-1,1]^2 → scale by half-size
    scale_vec = mi.ScalarVector3f(0.5 * sz.x, 0.5 * sz.y, 1.0)
    trans = trans @ T().scale(scale_vec)
    return trans

def gen_basis_from_z_dr(k: mi.Vector3f, eps: float = 1e-6):
    """
    Given a (vectorized) direction k (length = num_samples), build an
    orthonormal basis (u, v, k) with u ⟂ v ⟂ k. Numerically robust
    near the poles.
    Returns: (u, v) as Vector3f arrays (vectorized over samples).
    """
    up  = mi.Vector3f(0.0, 0.0, 1.0)
    alt = mi.Vector3f(0.0, 1.0, 0.0)
    # If k is too aligned with 'up', switch to 'alt' to avoid degeneracy
    safe = dr.select(dr.abs(k.z) > (1.0 - eps), alt, up)
    u = dr.normalize(dr.cross(safe, k))
    v = dr.normalize(dr.cross(k, u))
    return u, v


def ris_intersect(ris_objects, ray, active):
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
        indices_ = dr.reinterpret_array_v(mi.UInt32, si_ris.shape)

        valid |= v_
        new_closest = v_ & (t_ < t)

        t = dr.select(new_closest, t_, t)
        indices = dr.select(new_closest, indices_, indices)

    return valid, t, indices

def compute_spreading_factor(rho1: mi.Float, rho2: mi.Float, s: mi.Float) -> mi.Float:
    """
    sqrt( (rho1 * rho2) / ((rho1 + s) * (rho2 + s)) )
    with spherical case (rho1==0 and rho2==0) -> 1/s, using a safe reciprocal.
    """
    # Element-wise mask for spherical case
    spherical = (rho1 == 0.0) & (rho2 == 0.0)

    # Safe reciprocal: 1/s  (0 when s==0)
    inv_s = dr.select(s != 0.0, dr.rcp(s), mi.Float(0.0))

    # General case: avoid divide-by-zero and sqrt of tiny negatives
    denom = (rho1 + s) * (rho2 + s)
    ratio = safe_div(rho1 * rho2, denom)          # your existing per-lane safe divide
    ratio = dr.maximum(ratio, 0.0)                # clamp for numerical safety
    a = dr.sqrt(ratio)

    return dr.select(spherical, inv_s, a)

def safe_div(num: mi.Float, den: mi.Float) -> mi.Float:
    """Like tf.divide_no_nan: returns 0 where den == 0, otherwise num/den."""
    nonzero = (den != 0.0)              # per-lane mask
    return dr.select(nonzero, num * dr.rcp(den), mi.Float(0.0))

def safe_div_eps(num: mi.Float, den: mi.Float, eps: float = 0.0) -> mi.Float:
    mask = dr.abs(den) > eps
    return dr.select(mask, num * dr.rcp(den), mi.Float(0.0))


def safe_rcp(x: mi.Float) -> mi.Float:
    """Safe reciprocal: 0 where x == 0, otherwise 1/x."""
    nonzero = (x != 0.0)                # per-lane mask
    return dr.select(nonzero, dr.rcp(x), mi.Float(0.0))

def outer3(a: mi.Vector3f, b: mi.Vector3f) -> mi.Matrix3f:
    # 3x3 outer product, lane-wise
    return mi.Matrix3f(
        a.x*b.x, a.x*b.y, a.x*b.z,
        a.y*b.x, a.y*b.y, a.y*b.z,
        a.z*b.x, a.z*b.y, a.z*b.z
    )

def I3() -> mi.Matrix3f:
    return mi.Matrix3f(1.0)

def compute_spreading_factor_dr(rho1: mi.Float, rho2: mi.Float, s: mi.Float) -> mi.Float:
    # Same semantics as your TF version; spherical case => 1/s
    spherical = (rho1 == 0.0) & (rho2 == 0.0)
    a2_spherical = safe_rcp(s)
    a2 = dr.sqrt((rho1 * rho2) * safe_rcp((rho1 + s) * (rho2 + s)))
    return dr.select(spherical, a2_spherical, a2)

def gen_basis_from_k(k: mi.Vector3f) -> Tuple[mi.Vector3f, mi.Vector3f]:
    """Orthonormal (u,v) spanning plane ⟂ k, deterministic & stable."""
    # Choose a helper to avoid degeneracy
    sign = dr.copysign(1.0, k.z)
    a = -1.0 / (sign + k.z)
    b = k.x * k.y * a
    u = mi.Vector3f(1.0 + sign * k.x * k.x * a, sign * b, -sign * k.x)
    v = mi.Vector3f(b, sign + k.y * k.y * a, -k.y)
    # Normalize for safety
    u = dr.normalize(u); v = dr.normalize(v)
    return u, v

"""
def _top3_abs(x: mi.Float, supports: List[float]) -> Tuple[mi.UInt32, mi.UInt32, mi.UInt32,
                                                           mi.Float,  mi.Float,  mi.Float]:
    #Lane-wise 3 smallest |x - s_j| over a small list of scalar supports.
    B = dr.width(x)
    inf = float('inf')
    d1 = dr.full(mi.Float,  inf, B); i1 = dr.zeros(mi.UInt32, B); v1 = dr.zeros(mi.Float, B)
    d2 = dr.full(mi.Float,  inf, B); i2 = dr.zeros(mi.UInt32, B); v2 = dr.zeros(mi.Float, B)
    d3 = dr.full(mi.Float,  inf, B); i3 = dr.zeros(mi.UInt32, B); v3 = dr.zeros(mi.Float, B)

    for j, s in enumerate(supports):
        sj = mi.Float(float(s))
        dj = dr.abs(x - sj)

        b1 = dj < d1
        # shift 1->2, 2->3
        d3 = dr.select(b1, d2, d3); i3 = dr.select(b1, i2, i3); v3 = dr.select(b1, v2, v3)
        d2 = dr.select(b1, d1, d2); i2 = dr.select(b1, i1, i2); v2 = dr.select(b1, v1, v2)
        d1 = dr.select(b1, dj, d1); i1 = dr.select(b1, mi.UInt32(j), i1); v1 = dr.select(b1, sj, v1)

        b2 = (~b1) & (dj < d2)
        # shift 2->3
        d3 = dr.select(b2, d2, d3); i3 = dr.select(b2, i2, i3); v3 = dr.select(b2, v2, v3)
        d2 = dr.select(b2, dj, d2); i2 = dr.select(b2, mi.UInt32(j), i2); v2 = dr.select(b2, sj, v2)

        b3 = (~b1) & (~b2) & (dj < d3)
        d3 = dr.select(b3, dj, d3); i3 = dr.select(b3, mi.UInt32(j), i3); v3 = dr.select(b3, sj, v3)

    return i1, i2, i3, v1, v2, v3


def _lagrange3(y: mi.Float, a: mi.Float, b: mi.Float, c: mi.Float, return_grads=False):
    
    #3-point Lagrange basis at lane-wise y with nodes (a,b,c).
    #Returns:
    #  L0,L1,L2 (and optionally dL/dy, d2L/dy2).
    # denominators
    den0 = (a - b) * (a - c)
    den1 = (b - a) * (b - c)
    den2 = (c - a) * (c - b)

    # basis
    L0 = ((y - b) * (y - c)) * dr.rcp(den0)
    L1 = ((y - a) * (y - c)) * dr.rcp(den1)
    L2 = ((y - a) * (y - b)) * dr.rcp(den2)

    if not return_grads:
        return (L0, L1, L2)

    # first derivatives
    dL0 = (2.0 * y - (b + c)) * dr.rcp(den0)
    dL1 = (2.0 * y - (a + c)) * dr.rcp(den1)
    dL2 = (2.0 * y - (a + b)) * dr.rcp(den2)

    # second derivatives
    d2L0 = mi.Float(2.0) * dr.rcp(den0)
    d2L1 = mi.Float(2.0) * dr.rcp(den1)
    d2L2 = mi.Float(2.0) * dr.rcp(den2)

    return (L0, L1, L2), (dL0, dL1, dL2), (d2L0, d2L1, d2L2)
"""


def safe_rcp(x: mi.Float, eps: float = 0.0) -> mi.Float:
    # eps=0 -> exact check; raise eps if you want extra guarding
    return dr.select(dr.abs(x) > eps, dr.rcp(x), mi.Float(0.0))


def safe_div(num: mi.Float, den: mi.Float, eps: float = 0.0) -> mi.Float:
    return dr.select(dr.abs(den) > eps, num * dr.rcp(den), mi.Float(0.0))

def _top3_abs(x: mi.Float, supports)-> tuple[mi.UInt32, mi.UInt32, mi.UInt32,
                                              mi.Float,  mi.Float,  mi.Float]:
    """Lane-wise 3 smallest |x - s_j|; returns (idx1,idx2,idx3, a,b,c) with a,b,c the *coordinates*."""
    B = dr.width(x)
    inf = float('inf')

    d1 = dr.full(mi.Float,  inf, B); i1 = dr.zeros(mi.UInt32, B); v1 = dr.zeros(mi.Float, B)
    d2 = dr.full(mi.Float,  inf, B); i2 = dr.zeros(mi.UInt32, B); v2 = dr.zeros(mi.Float, B)
    d3 = dr.full(mi.Float,  inf, B); i3 = dr.zeros(mi.UInt32, B); v3 = dr.zeros(mi.Float, B)

    for j, s in enumerate(supports):
        sj = mi.Float(float(s))
        dj = dr.abs(x - sj)

        b1 = dj < d1
        d3 = dr.select(b1, d2, d3); i3 = dr.select(b1, i2, i3); v3 = dr.select(b1, v2, v3)
        d2 = dr.select(b1, d1, d2); i2 = dr.select(b1, i1, i2); v2 = dr.select(b1, v1, v2)
        d1 = dr.select(b1, dj, d1); i1 = dr.select(b1, mi.UInt32(j), i1); v1 = dr.select(b1, sj, v1)

        b2 = (~b1) & (dj < d2)
        d3 = dr.select(b2, d2, d3); i3 = dr.select(b2, i2, i3); v3 = dr.select(b2, v2, v3)
        d2 = dr.select(b2, dj, d2); i2 = dr.select(b2, mi.UInt32(j), i2); v2 = dr.select(b2, sj, v2)

        b3 = (~b1) & (~b2) & (dj < d3)
        d3 = dr.select(b3, dj, d3); i3 = dr.select(b3, mi.UInt32(j), i3); v3 = dr.select(b3, sj, v3)

    return i1, i2, i3, v1, v2, v3

"""
def _lagrange3(y: mi.Float, a: mi.Float, b: mi.Float, c: mi.Float,
               return_grads: bool = False, eps: float = 1e-8):
    Robust 3-point Lagrange at lane-wise y with nodes (a,b,c).
    Snaps when y~node, avoids 0/0 when nodes are (nearly) coincident, and can return d/dy, d2/dy2.

    # Snap to node to avoid 0/0 & keep gradients finite
    n0 = dr.abs(y - a) < eps
    n1 = dr.abs(y - b) < eps
    n2 = dr.abs(y - c) < eps
    snap = n0 | n1 | n2

    L0 = dr.select(n0, mi.Float(1.0), mi.Float(0.0))
    L1 = dr.select(n1, mi.Float(1.0), mi.Float(0.0))
    L2 = dr.select(n2, mi.Float(1.0), mi.Float(0.0))

    # Ill-conditioned nodes (nearly coincident)
    ill = (dr.abs(a - b) < eps) | (dr.abs(a - c) < eps) | (dr.abs(b - c) < eps)

    # Regular path where not snapped and nodes are well-separated
    den0 = (a - b) * (a - c)
    den1 = (b - a) * (b - c)
    den2 = (c - a) * (c - b)

    t0 = ((y - b) * (y - c)) * safe_rcp(den0)
    t1 = ((y - a) * (y - c)) * safe_rcp(den1)
    t2 = ((y - a) * (y - b)) * safe_rcp(den2)

    L0 = dr.select(~snap & ~ill, t0, L0)
    L1 = dr.select(~snap & ~ill, t1, L1)
    L2 = dr.select(~snap & ~ill, t2, L2)

    # Optional: renormalize to sum=1 (helps numerics)
    S = L0 + L1 + L2
    invS = safe_div(mi.Float(1.0), S, eps=0.0)
    L0 = L0 * invS; L1 = L1 * invS; L2 = L2 * invS

    if not return_grads:
        return (L0, L1, L2)

    # Derivatives: zero where snapped or ill; regular formula elsewhere
    dL0 = mi.Float(0.0); dL1 = mi.Float(0.0); dL2 = mi.Float(0.0)
    d2L0 = mi.Float(0.0); d2L1 = mi.Float(0.0); d2L2 = mi.Float(0.0)

    dt0  = (mi.Float(2.0) * y - (b + c)) * safe_rcp(den0)
    dt1  = (mi.Float(2.0) * y - (a + c)) * safe_rcp(den1)
    dt2  = (mi.Float(2.0) * y - (a + b)) * safe_rcp(den2)

    d2t0 = mi.Float(2.0) * safe_rcp(den0)
    d2t1 = mi.Float(2.0) * safe_rcp(den1)
    d2t2 = mi.Float(2.0) * safe_rcp(den2)

    reg = ~snap & ~ill
    dL0 = dr.select(reg, dt0, dL0); dL1 = dr.select(reg, dt1, dL1); dL2 = dr.select(reg, dt2, dL2)
    d2L0 = dr.select(reg, d2t0, d2L0); d2L1 = dr.select(reg, d2t1, d2L1); d2L2 = dr.select(reg, d2t2, d2L2)

    return (L0, L1, L2), (dL0, dL1, dL2), (d2L0, d2L1, d2L2)
"""

def _lagrange3(y: mi.Float, a: mi.Float, b: mi.Float, c: mi.Float, return_grads=False): 
    """ 
    3-point Lagrange basis at lane-wise y with nodes (a,b,c). 
    Returns: L0,L1,L2 (and optionally dL/dy, d2L/dy2). 
    """ 
    # denominators 
    den0 = (a - b) * (a - c) 
    den1 = (b - a) * (b - c) 
    den2 = (c - a) * (c - b) 
    
    # basis 
    print(y.shape, b.shape)
    L0 = ((y - b) * (y - c)) * dr.rcp(den0) 
    L1 = ((y - a) * (y - c)) * dr.rcp(den1) 
    L2 = ((y - a) * (y - b)) * dr.rcp(den2) 


    
    if not return_grads: 
        return (L0, L1, L2) 
    
    # first derivatives 
    dL0 = (2.0 * y - (b + c)) * dr.rcp(den0) 
    dL1 = (2.0 * y - (a + c)) * dr.rcp(den1) 
    dL2 = (2.0 * y - (a + b)) * dr.rcp(den2)
    
    # second derivatives 
    d2L0 = mi.Float(2.0) * dr.rcp(den0) 
    d2L1 = mi.Float(2.0) * dr.rcp(den1) 
    d2L2 = mi.Float(2.0) * dr.rcp(den2) 
    
    return (L0, L1, L2), (dL0, dL1, dL2), (d2L0, d2L1, d2L2)