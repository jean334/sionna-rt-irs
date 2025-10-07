# SPDX-FileCopyrightText: Copyright (c) 2025 Jean ACKER.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple
import tensorflow as tf
import mitsuba as mi
import drjit as dr
from typing_extensions import Tuple
import irs.config_legacy
from rt.utils.irs_tensors import expand_to_rank, expand_to_rank_dr
from rt.utils.irs_misc import log10
from numpy import pi as PI

config = irs.config_legacy.Config()

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


def dr_squeeze(x):
    """Remove dimensions of size 1 (like tf.squeeze)."""
    shape = tuple(d for d in x.shape if d != 1)
    if not shape:
        shape = (1,)   # keep at least a scalar
    return dr.reshape(dr.cuda.ad.TensorXf, x, shape)


def rotation_matrix_dr(angles, inverse=True):
    r"""
    Computes rotation matrices as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) [TR38901]_ is used.

    Input
    ------
    angles : [3,...], dr.cuda.ad.TensorXf
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    Output
    -------
    :  [3,3,...], dr.cuda.TensorXf
        Rotation matrices
    """
    if(len(angles.shape) >= 2):
        a = angles[0,...]
        b = angles[1,...]
        c = angles[2,...]
    elif(len(angles.shape) == 1):
        a = angles[0]
        b = angles[1]
        c = angles[2]
    cos_a = dr.cos(a)
    cos_b = dr.cos(b)
    cos_c = dr.cos(c)
    sin_a = dr.sin(a)
    sin_b = dr.sin(b)
    sin_c = dr.sin(c)
    r_11 = dr.cuda.ad.Float(cos_a*cos_b)
    r_12 = dr.cuda.ad.Float(cos_a*sin_b*sin_c - sin_a*cos_c)
    r_13 = dr.cuda.ad.Float(cos_a*sin_b*cos_c + sin_a*sin_c)
    
    r_21 = dr.cuda.ad.Float(sin_a*cos_b)
    r_22 = dr.cuda.ad.Float(sin_a*sin_b*sin_c + cos_a*cos_c)
    r_23 = dr.cuda.ad.Float(sin_a*sin_b*cos_c - cos_a*sin_c)

    r_31 = dr.cuda.ad.Float(-sin_b)
    r_32 = dr.cuda.ad.Float(cos_b*sin_c)
    r_33 = dr.cuda.ad.Float(cos_b*cos_c)

    if inverse==True:
        #[3,3,-1]
        rot_mat = dr.empty(dr.cuda.ad.TensorXf, (3, 3,  angles.shape[-1]))
        rot_mat[0,0,:] = r_11
        rot_mat[0,1,:] = r_12
        rot_mat[0,2,:] = r_13
        rot_mat[1,0,:] = r_21
        rot_mat[1,1,:] = r_22
        rot_mat[1,2,:] = r_23
        rot_mat[2,0,:] = r_31
        rot_mat[2,1,:] = r_32
        rot_mat[2,2,:] = r_33
    else:
        #[-1, 3,3]
        rot_mat = dr.empty(dr.cuda.ad.TensorXf, (3, 3, angles.shape[-1]))
        rot_mat[0,0,:] = r_11
        rot_mat[1,0,:] = r_12
        rot_mat[2,0,:] = r_13
        rot_mat[0,1,:] = r_21
        rot_mat[1,1,:] = r_22
        rot_mat[2,1,:] = r_23
        rot_mat[0,2,:] = r_31
        rot_mat[1,2,:] = r_32
        rot_mat[2,2,:] = r_33

    return rot_mat




def rotation_matrix(angles):
    r"""
    Computes rotation matrices as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) [TR38901]_ is used.

    Input
    ------
    angles : [...,3], tf.float
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    Output
    -------
    : [...,3,3], tf.float
        Rotation matrices
    """

    a = angles[...,0]
    b = angles[...,1]
    c = angles[...,2]
    cos_a = tf.cos(a)
    cos_b = tf.cos(b)
    cos_c = tf.cos(c)
    sin_a = tf.sin(a)
    sin_b = tf.sin(b)
    sin_c = tf.sin(c)

    r_11 = cos_a*cos_b
    r_12 = cos_a*sin_b*sin_c - sin_a*cos_c
    r_13 = cos_a*sin_b*cos_c + sin_a*sin_c
    r_1 = tf.stack([r_11, r_12, r_13], axis=-1)

    r_21 = sin_a*cos_b
    r_22 = sin_a*sin_b*sin_c + cos_a*cos_c
    r_23 = sin_a*sin_b*cos_c - cos_a*sin_c
    r_2 = tf.stack([r_21, r_22, r_23], axis=-1)

    r_31 = -sin_b
    r_32 = cos_b*sin_c
    r_33 = cos_b*cos_c
    r_3 = tf.stack([r_31, r_32, r_33], axis=-1)

    rot_mat = tf.stack([r_1, r_2, r_3], axis=-2)
    return rot_mat

def rotate_dr(p, angles, inverse=False):
    r"""
    Rotates points ``p`` by the ``angles`` according
    to the 3D rotation defined in :eq:`rotation`

    Input
    -----
    p : [3,...], tf.float
        Points to rotate

    angles : [3,...]
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    inverse : bool
        If `True`, the inverse rotation is applied,
        i.e., the transpose of the rotation matrix is used.
        Defaults to `False`

    Output
    ------
    : [3,...]
        Rotated points ``p``
    """

    # Rotation matrix
    # [3,3, ...]
    rot_mat = rotation_matrix_dr(angles)
    rot_mat = expand_to_rank_dr(rot_mat, len(p.shape)+1, 0)

    # Rotation around ``center``
    # [3,2]*[3,3,2] -->  [3,2]
    if inverse==True:
        rot_p = dr.dot(rot_mat, p)
    elif inverse==False:
        rot_p = dr.dot(rot_mat,p)

    return rot_p

def rotate(p, angles, inverse=False):
    r"""
    Rotates points ``p`` by the ``angles`` according
    to the 3D rotation defined in :eq:`rotation`

    Input
    -----
    p : [...,3], tf.float
        Points to rotate

    angles : [..., 3]
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    inverse : bool
        If `True`, the inverse rotation is applied,
        i.e., the transpose of the rotation matrix is used.
        Defaults to `False`

    Output
    ------
    : [...,3]
        Rotated points ``p``
    """

    # Rotation matrix
    # [..., 3, 3]

    rot_mat = rotation_matrix(angles)
    rot_mat = expand_to_rank(rot_mat, tf.rank(p)+1, 0)

    # Rotation around ``center``
    # [..., 3]
    rot_p = tf.linalg.matvec(rot_mat, p, transpose_a=inverse)

    return rot_p

def theta_phi_from_unit_vec(v):
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    Input
    ------
    v : [...,3], tf.float
        Tensor with unit-norm vectors in the last dimension

    Output
    -------
    theta : [...], tf.float
        Zenith angles :math:`\theta`

    phi : [...], tf.float
        Azimuth angles :math:`\varphi`
    """
    x = v[...,0]
    y = v[...,1]
    z = v[...,2]

    # If v = z, then x = 0 and y = 0. In this case, atan2 is not differentiable,
    # leading to NaN when computing the gradients.
    # The following lines force x to one this case. Note that this does not
    # impact the output meaningfully, as in that case theta = 0 and phi can
    # take any value.
    zero = tf.zeros_like(x)
    is_unit_z = tf.logical_and(tf.equal(x, zero), tf.equal(y, zero))
    is_unit_z = tf.cast(is_unit_z, x.dtype)
    x += is_unit_z

    theta = acos_diff(z)
    phi = tf.math.atan2(y, x)
    return theta, phi

def r_hat(theta, phi):
    r"""
    Computes the spherical unit vetor :math:`\hat{\mathbf{r}}(\theta, \phi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    theta : arbitrary shape, tf.float
        Zenith angles :math:`\theta` [rad]

    phi : same shape as ``theta``, tf.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    rho_hat : ``phi.shape`` + [3], tf.float
        Vector :math:`\hat{\mathbf{r}}(\theta, \phi)`  on unit sphere
    """
    rho_hat = tf.stack([tf.sin(theta)*tf.cos(phi),
                        tf.sin(theta)*tf.sin(phi),
                        tf.cos(theta)], axis=-1)
    return rho_hat

def theta_hat(theta, phi):
    r"""
    Computes the spherical unit vector
    :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    theta : arbitrary shape, tf.float
        Zenith angles :math:`\theta` [rad]

    phi : same shape as ``theta``, tf.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    theta_hat : ``phi.shape`` + [3], tf.float
        Vector :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    """
    x = tf.cos(theta)*tf.cos(phi)
    y = tf.cos(theta)*tf.sin(phi)
    z = -tf.sin(theta)
    return tf.stack([x,y,z], -1)

def phi_hat(phi):
    r"""
    Computes the spherical unit vector
    :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    phi : same shape as ``theta``, tf.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    theta_hat : ``phi.shape`` + [3], tf.float
        Vector :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    """
    x = -tf.sin(phi)
    y = tf.cos(phi)
    z = tf.zeros_like(x)
    return tf.stack([x,y,z], -1)

def cross(u, v):
    r"""
    Computes the cross (or vector) product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    Output
    -------
    : [...,3]
        Cross product between ``u`` and ``v``
    """
    u_x = u[...,0]
    u_y = u[...,1]
    u_z = u[...,2]

    v_x = v[...,0]
    v_y = v[...,1]
    v_z = v[...,2]

    w = tf.stack([u_y*v_z - u_z*v_y,
                  u_z*v_x - u_x*v_z,
                  u_x*v_y - u_y*v_x], axis=-1)

    return w

def dot(u, v, keepdim=False, clip=False):
    r"""
    Computes and the dot (or scalar) product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    keepdim : bool
        If `True`, keep the last dimension.
        Defaults to `False`.

    clip : bool
        If `True`, clip output to [-1,1].
        Defaults to `False`.

    Output
    -------
    : [...,1] or [...]
        Dot product between ``u`` and ``v``.
        The last dimension is removed if ``keepdim``
        is set to `False`.
    """
    res = tf.reduce_sum(u*v, axis=-1, keepdims=keepdim)
    if clip:
        one = tf.ones((), u.dtype)
        res = tf.clip_by_value(res, -one, one)
    return res

def outer(u,v):
    r"""
    Computes the outer product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    Output
    -------
    : [...,3,3]
        Outer product between ``u`` and ``v``
    """
    return u[...,tf.newaxis] * v[...,tf.newaxis,:]

def outer_dr(u,v):
    return u[dr.newaxis] * v[:,dr.newaxis,...]

def normalize(v):
    r"""
    Normalizes ``v`` to unit norm

    Input
    ------
    v : [...,3], tf.float
        Vector

    Output
    -------
    : [...,3], tf.float
        Normalized vector

    : [...], tf.float
        Norm of the unnormalized vector
    """
    norm = tf.norm(v, axis=-1, keepdims=True)
    n_v = tf.math.divide_no_nan(v, norm)
    norm = tf.squeeze(norm, axis=-1)
    return n_v, norm

def normalize_dr(v):
    r"""
    Normalizes ``v`` to unit norm

    Input
    ------
    v : [3,...], dr.cuda.ad.TensorXf
        Vector

    Output
    -------
    : [3,...], dr.cuda.ad.TensorXf
        Normalized vector

    : [...], dr.cuda.ad.TensorXf
        Norm of the unnormalized vector
    """
    norm = dr.norm(v)
    nan_val = dr.isnan(norm)
    norm = dr.select(nan_val, 1, norm)
    n_v = v/norm
    #norm = dr_squeeze(norm)

    return n_v, norm

def moller_trumbore(o, d, p0, p1, p2, epsilon):
    r"""
    Computes the intersection between a ray ``ray`` and a triangle defined
    by its vertices ``p0``, ``p1``, and ``p2`` using the Moller–Trumbore
    intersection algorithm.

    Input
    -----
    o, d: [..., 3], tf.float
        Ray origin and direction.
        The direction `d` must be a unit vector.

    p0, p1, p2: [..., 3], tf.float
        Vertices defining the triangle

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    t : [...], tf.float
        Position along the ray from the origin at which the intersection
        occurs (if any)

    hit : [...], bool
        `True` if the ray intersects the triangle. `False` otherwise.
    """

    rdtype = o.dtype
    zero = tf.cast(0.0, rdtype)
    one = tf.ones((), rdtype)

    # [..., 3]
    e1 = p1 - p0
    e2 = p2 - p0

    # [...,3]
    pvec = cross(d, e2)
    # [...,1]
    det = dot(e1, pvec, keepdim=True)

    # If the ray is parallel to the triangle, then det = 0.
    hit = tf.greater(tf.abs(det), zero)

    # [...,3]
    tvec = o - p0
    # [...,1]
    u = tf.math.divide_no_nan(dot(tvec, pvec, keepdim=True), det)
    # [...,1]
    hit = tf.logical_and(hit,
        tf.logical_and(tf.greater_equal(u, -epsilon),
                       tf.less_equal(u, one + epsilon)))

    # [..., 3]
    qvec = cross(tvec, e1)
    # [...,1]
    v = tf.math.divide_no_nan(dot(d, qvec, keepdim=True), det)
    # [..., 1]
    hit = tf.logical_and(hit,
                            tf.logical_and(tf.greater_equal(v, -epsilon),
                                        tf.less_equal(u + v, one + epsilon)))
    # [..., 1]
    t = tf.math.divide_no_nan(dot(e2, qvec, keepdim=True), det)
    # [..., 1]
    hit = tf.logical_and(hit, tf.greater_equal(t, epsilon))

    # [...]
    t = tf.squeeze(t, axis=-1)
    hit = tf.squeeze(hit, axis=-1)

    return t, hit

def component_transform(e_s, e_p, e_i_s, e_i_p):
    """
    Compute basis change matrix for reflections

    Input
    -----
    e_s : [..., 3], tf.float
        Source unit vector for S polarization

    e_p : [..., 3], tf.float
        Source unit vector for P polarization

    e_i_s : [..., 3], tf.float
        Target unit vector for S polarization

    e_i_p : [..., 3], tf.float
        Target unit vector for P polarization

    Output
    -------
    r : [..., 2, 2], tf.float
        Change of basis matrix for going from (e_s, e_p) to (e_i_s, e_i_p)
    """
    r_11 = dot(e_i_s, e_s)
    r_12 = dot(e_i_s, e_p)
    r_21 = dot(e_i_p, e_s)
    r_22 = dot(e_i_p, e_p)
    r1 = tf.stack([r_11, r_12], axis=-1)
    r2 = tf.stack([r_21, r_22], axis=-1)
    r = tf.stack([r1, r2], axis=-2)
    return r

def mi_to_tf_tensor(mi_tensor, dtype):
    """
    Get a TensorFlow eager tensor from a Mitsuba/DrJIT tensor
    """
    dr.eval(mi_tensor)
    dr.sync_thread()
    # When there is only one input, the .tf() methods crashes.
    # The following hack takes care of this corner case
    if dr.shape(mi_tensor)[-1] == 1:
        mi_tensor = dr.repeat(mi_tensor, 2)
        tf_tensor = tf.cast(mi_tensor.tf(), dtype)[:1]
    else:
        tf_tensor = tf.cast(mi_tensor.tf(), dtype)
    return tf_tensor

def gen_orthogonal_vector(k, epsilon):
    """
    Generate an arbitrary vector that is orthogonal to ``k``.

    Input
    ------
    k : [..., 3], tf.float
        Vector

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    : [..., 3], tf.float
        Vector orthogonal to ``k``
    """
    rdtype = k.dtype
    ex = tf.cast([1.0, 0.0, 0.0], rdtype)
    ex = expand_to_rank(ex, tf.rank(k), 0)

    ey = tf.cast([0.0, 1.0, 0.0], rdtype)
    ey = expand_to_rank(ey, tf.rank(k), 0)

    n1 = cross(k, ex)
    n1_norm = tf.norm(n1, axis=-1, keepdims=True)
    n2 = cross(k, ey)
    return tf.where(tf.greater(n1_norm, epsilon), n1, n2)

def compute_field_unit_vectors(k_i, k_r, n, epsilon, return_e_r=True):
    """
    Compute unit vector parallel and orthogonal to incident plane

    Input
    ------
    k_i : [..., 3], tf.float
        Direction of arrival

    k_r : [..., 3], tf.float
        Direction of reflection

    n : [..., 3], tf.float
        Surface normal

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    return_e_r : bool
        If `False`, only ``e_i_s`` and ``e_i_p`` are returned.

    Output
    ------
    e_i_s : [..., 3], tf.float
        Incident unit field vector for S polarization

    e_i_p : [..., 3], tf.float
        Incident unit field vector for P polarization

    e_r_s : [..., 3], tf.float
        Reflection unit field vector for S polarization.
        Only returned if ``return_e_r`` is `True`.

    e_r_p : [..., 3], tf.float
        Reflection unit field vector for P polarization
        Only returned if ``return_e_r`` is `True`.
    """
    e_i_s = cross(k_i, n)
    e_i_s_norm = tf.norm(e_i_s, axis=-1, keepdims=True)
    # In case of normal incidence, the incidence plan is not uniquely
    # define and the Fresnel coefficent is the same for both polarization
    # (up to a sign flip for the parallel component due to the definition of
    # polarization).
    # It is required to detect such scenarios and define an arbitrary valid
    # e_i_s to fix an incidence plane, as the result from previous
    # computation leads to e_i_s = 0.
    e_i_s = tf.where(tf.greater(e_i_s_norm, epsilon), e_i_s,
                     gen_orthogonal_vector(n, epsilon))

    e_i_s,_ = normalize(e_i_s)
    e_i_p,_ = normalize(cross(e_i_s, k_i))
    if not return_e_r:
        return e_i_s, e_i_p
    else:
        e_r_s = e_i_s
        e_r_p,_ = normalize(cross(e_r_s, k_r))
        return e_i_s, e_i_p, e_r_s, e_r_p

def reflection_coefficient(eta, cos_theta):
    """
    Compute simplified reflection coefficients

    Input
    ------
    eta : Any shape, tf.complex
        Complex relative permittivity

    cos_theta : Same as ``eta``, tf.float
        Cosine of the incident angle

    Output
    -------
    r_te : Same as input, tf.complex
        Fresnel reflection coefficient for S direction

    r_tm : Same as input, tf.complex
        Fresnel reflection coefficient for P direction
    """
    cos_theta = tf.complex(cos_theta, tf.zeros_like(cos_theta))

    # Fresnel equations
    a = cos_theta
    b = tf.sqrt(eta-1.+cos_theta**2)
    r_te = tf.math.divide_no_nan(a-b, a+b)

    c = eta*a
    d = b
    r_tm = tf.math.divide_no_nan(c-d, c+d)
    return r_te, r_tm

def paths_to_segments(paths):
    """
    Extract the segments corresponding to a set of ``paths``

    Input
    -----
    paths : :class:`~sionna.rt.Paths`
        A set of paths

    Output
    -------
    starts, ends : [n,3], float
        Endpoints of the segments making the paths.
    """

    vertices = paths.vertices.numpy()
    objects = paths.objects.numpy()
    mask = paths.targets_sources_mask
    sources, targets = paths.sources.numpy(), paths.targets.numpy()

    # Emit directly two lists of the beginnings and endings of line segments
    starts = []
    ends = []
    for rx in range(vertices.shape[1]): # For each receiver
        for tx in range(vertices.shape[2]): # For each transmitter
            for p in range(vertices.shape[3]): # For each path depth
                if not mask[rx, tx, p]:
                    continue

                start = sources[tx]
                i = 0
                while ( (i < objects.shape[0])
                    and (objects[i, rx, tx, p] != -1) ):
                    end = vertices[i, rx, tx, p]
                    starts.append(start)
                    ends.append(end)
                    start = end
                    i += 1
                # Explicitly add the path endpoint
                starts.append(start)
                ends.append(targets[rx])
    return starts, ends

def scene_scale(scene):
    bbox = scene.mi_scene.bbox()
    tx_positions, rx_positions, ris_positions = {}, {}, {}
    devices = ((scene.transmitters, tx_positions),
               (scene.receivers, rx_positions),
               (scene.ris, ris_positions)
              )
    for source, destination in devices:
        for k, rd in source.items():
            p = rd.position.numpy()
            bbox.expand(p)
            destination[k] = p

    sc = 2. * bbox.bounding_sphere().radius
    return sc, tx_positions, rx_positions, ris_positions, bbox

def fibonacci_lattice(num_points, dtype=tf.float32):
    """
    Generates a Fibonacci lattice for the unit square

    Input
    -----
    num_points : int
        Number of points

    type : tf.DType
        Datatype to use for the output

    Output
    -------
    points : [num_points, 2]
        Generated rectangular coordinates of the lattice points
    """

    golden_ratio = (1.+tf.sqrt(tf.cast(5., tf.float64)))/2.
    ns = tf.range(0, num_points, dtype=tf.float64)

    x = ns/golden_ratio
    x = x - tf.floor(x)
    y = ns/(num_points-1)
    points = tf.stack([x,y], axis=1)

    points = tf.cast(points, dtype)

    return points

def cot(x):
    """
    Cotangens function

    Input
    ------
    x : [...], tf.float

    Output
    -------
    : [...], tf.float
        Cotangent of x
    """
    return tf.math.divide_no_nan(tf.ones_like(x), tf.math.tan(x))

def sign(x):
    """
    Returns +1 if ``x`` is non-negative, -1 otherwise

    Input
    ------
    x : [...], tf.float
        A real-valued number

    Output
    -------
    : [...], tf.float
        +1 if ``x`` is non-negative, -1 otherwise
    """
    two = tf.cast(2, x.dtype)
    one = tf.cast(1, x.dtype)
    return two*tf.cast(tf.greater_equal(x, 0), x.dtype)-one

def rot_mat_from_unit_vecs(a, b):
    r"""
    Computes Rodrigues` rotation formula :eq:`rodrigues_matrix`

    Input
    ------
    a : [...,3], tf.float
        First unit vector

    b : [...,3], tf.float
        Second unit vector

    Output
    -------
    : [...,3,3], tf.float
        Rodrigues' rotation matrix
    """

    rdtype = a.dtype

    # Compute rotation axis vector
    k, _ = normalize(cross(a, b))

    # Deal with special case where a and b are parallel
    o = gen_orthogonal_vector(a, 1e-6)
    k = tf.where(tf.reduce_sum(tf.abs(k), axis=-1, keepdims=True)==0, o, k)

    # Compute K matrix
    shape = tf.concat([tf.shape(k)[:-1],[1]], axis=-1)
    zeros = tf.zeros(shape, rdtype)
    kx, ky, kz = tf.split(k, 3, axis=-1)
    l1 = tf.concat([zeros, -kz, ky], axis=-1)
    l2 = tf.concat([kz, zeros, -kx], axis=-1)
    l3 = tf.concat([-ky, kx, zeros], axis=-1)
    k_mat = tf.stack([l1, l2, l3], axis=-2)

    # Assemble full rotation matrix
    eye = tf.eye(3, batch_shape=tf.shape(k)[:-1], dtype=rdtype)
    cos_theta = dot(a, b, clip=True)
    sin_theta = tf.sin(acos_diff(cos_theta))
    cos_theta = expand_to_rank(cos_theta, tf.rank(eye), axis=-1)
    sin_theta = expand_to_rank(sin_theta, tf.rank(eye), axis=-1)
    rot_mat = eye + k_mat*sin_theta + \
                      tf.linalg.matmul(k_mat, k_mat) * (1-cos_theta)
    return rot_mat

def sample_points_on_hemisphere(normals, num_samples=1):
    # pylint: disable=line-too-long
    r"""
    Randomly sample points on hemispheres defined by their normal vectors

    Input
    -----
    normals : [batch_size, 3], tf.float
        Normal vectors defining hemispheres

    num_samples : int
        Number of random samples to draw for each hemisphere
        defined by its normal vector.
        Defaults to 1.

    Output
    ------
    points : [batch_size, num_samples, 3], tf.float or [batch_size, 3], tf.float if num_samples=1.
        Random points on the hemispheres
    """
    dtype = normals.dtype
    batch_size = tf.shape(normals)[0]
    shape = [batch_size, num_samples]

    # Sample phi uniformly distributed on [0,2*PI]
    phi = config.tf_rng.uniform(shape, maxval=2*PI, dtype=dtype)

    # Generate samples of theta for uniform distribution on the hemisphere
    u = config.tf_rng.uniform(shape, maxval=1, dtype=dtype)
    theta = tf.acos(u)

    # Transform spherical to Cartesian coordinates
    points = r_hat(theta, phi)

    # Compute rotation matrices
    z_hat = tf.constant([[0,0,1]], dtype=dtype)
    z_hat = tf.broadcast_to(z_hat, tf.shape(normals))
    rot_mat = rot_mat_from_unit_vecs(z_hat, normals)
    rot_mat = tf.expand_dims(rot_mat, axis=1)

    # Compute rotated points
    points = tf.linalg.matvec(rot_mat, points)

    # Numerical errors can cause sampling from the other hemisphere.
    # Correct the sampled vector to avoid sampling in the wrong hemisphere.
    normals = tf.expand_dims(normals, axis=1)
    s = dot(points, normals, keepdim=True)
    s = tf.where(s < 0., s, 0.)
    points = points - 2.*s*normals

    if num_samples==1:
        points = tf.squeeze(points, axis=1)

    return points

def acos_diff(x, epsilon=1e-7):
    r"""
    Implementation of arccos(x) that avoids evaluating the gradient at x
    -1 or 1 by using straight through estimation, i.e., in the
    forward pass, x is clipped to (-1, 1), but in the backward pass, x is
    clipped to (-1 + epsilon, 1 - epsilon).

    Input
    ------
    x : any shape, tf.float
        Value at which to evaluate arccos

    epsilon : tf.float
        Small backoff to avoid evaluating the gradient at -1 or 1.
        Defaults to 1e-7.

    Output
    -------
     : same shape as x, tf.float
        arccos(x)
    """

    x_clip_1 = tf.clip_by_value(x, -1., 1.)
    x_clip_2 = tf.clip_by_value(x, -1. + epsilon, 1. - epsilon)
    eps = tf.stop_gradient(x - x_clip_2)
    x_1 =  x - eps
    acos_x_1 =  tf.acos(x_1)
    y = acos_x_1 + tf.stop_gradient(tf.acos(x_clip_1)-acos_x_1)
    return y


def angles_to_mitsuba_rotation(angles):
    #Build a Mitsuba transform from angles in radian

    #Input
    #------
    #angles : [3], tf.float
    #    Angles [rad]

    #Output
    #-------
    #: :class:`mitsuba.ScalarTransform4f`
    #    Mitsuba rotation
    

    angles = 180. * angles / PI

    if angles.dtype == tf.float32:
        mi_transform_t = mi.ScalarTransform4f
        angles = mi.Float(angles)
    else:
        mi_transform_t = mi.ScalarTransform4d
        angles = mi.Float64(angles)

    return (
          mi.Transform4f().rotate(axis=[0, 0, 1], angle=angles[0])
        @ mi.Transform4f().rotate(axis=[0, 1, 0], angle=angles[1])
        @ mi.Transform4f().rotate(axis=[1, 0, 0], angle=angles[2])
    )


def gen_basis_from_z(z, epsilon):
    """
    Generate a pair of vectors (x,y) such that (x,y,z) is an orthonormal basis.

    Input
    ------
    z : [..., 3], tf.float
        Unit vector

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    x : [..., 3], tf.float
        Unit vector

    y : [..., 3], tf.float
        Unit vector
    """
    x = gen_orthogonal_vector(z, epsilon)
    x,_ = normalize(x)
    y = cross(z, x)
    return x,y

def compute_spreading_factor(rho_1, rho_2, s):
    r"""
    Computes the spreading factor
    :math:`\sqrt{\frac{\rho_1 \rho_2}{(\rho_1 + s)(\rho_2 + s)}}`

    Input
    ------
    rho_1, rho_2 : [...], tf.float
        Principal radii of curvature

    s : [...], tf.float
        Position along the axial ray at which to evaluate the squared
        spreading factor

    Output
    -------
    : float
        Squared spreading factor
    """

    # In the case of a spherical wave, when the origin (s = 0) is set to unique
    # caustic point, then both principal radii of curvature are set to zero.
    # The spreading factor is then equal to 1/s.
    spherical = tf.logical_and(tf.equal(rho_1, 0.), tf.equal(rho_2, 0.))
    a2_spherical = tf.math.reciprocal_no_nan(s)

    # General formula for the spreading factor
    a2 = tf.sqrt(rho_1*rho_2/((rho_1+s)*(rho_2+s)))

    a2 = tf.where(spherical, a2_spherical, a2)
    return a2

"""
def mitsuba_rectangle_to_world(center, orientation, size, ris=False):
    Build the `to_world` transformation that maps a default Mitsuba rectangle
    to the rectangle that defines the coverage map surface.

    Input
    ------
    center : [3], tf.float
        Center of the rectangle

    orientation : [3], tf.float
        Orientation of the rectangle.
        An orientation of `(0,0,0)` correspond to a rectangle oriented such that
        z+ is its normal.

    size : [2], tf.float
        Scale of the rectangle.
        The width of the rectangle (in the local X direction) is scale[0]
        and its height (in the local Y direction) scale[1].

    Output
    -------
    to_world : :class:`mitsuba.ScalarTransform4f`
        Rectangle to world transformation.
    
    orientation = 180. * orientation / PI

    trans = \
        mi.ScalarTransform4f().translate(mi.ScalarPoint3f(center.numpy()))\
        @ mi.ScalarTransform4f().rotate(axis=mi.ScalarPoint3f([0, 0, 1]), angle=orientation[0])\
        @ mi.ScalarTransform4f().rotate(axis=mi.ScalarPoint3f([0, 1, 0]), angle=orientation[1])\
        @ mi.ScalarTransform4f().rotate(axis=mi.ScalarPoint3f([1, 0, 0]), angle=orientation[2])

    if ris:
        # The RIS normal points at [1,0,0].
        # We hence rotate the normal of the rectangle which points
        # at [0,0,1] by 90 degrees around the [0,1,0] axis.
        trans = trans\
            @mi.ScalarTransform4f().rotate(axis=mi.ScalarPoint3f([0, 1, 0]), angle=90)

        # size = [width (=y), height (=z)]
        # Since the RIS is rotated w.r.t to rectangle,
        # The z axis corresponds to the x axis
        size = [size[1], size[0]]

    return (trans
            @mi.ScalarTransform4f().scale(mi.ScalarPoint3f([0.5 * size[0], 0.5 * size[1], 1]))
    )
"""

def watt_to_dbm(power):
    r""" Converts :math:`P_{W}` [W] to :math:`P_{dBm}` [dBm] via the formula:
    :math:`P_{dBm} = 30 + 10 \log_{10}(P_W)`

    Input
    ------
    power : float
        Power [W]

    Output
    -------
     : float
        Power [dBm]
    """
    return 30 + 10 * log10(power)

def dbm_to_watt(dbm):
    r""" Converts dBm to Watt via the formula:
    :math:`P_W = 10^{\frac{P_{dBm}-30}{10}}`

    Input
    ------
    dbm : float
        Power [dBm]

    Output
    -------
     : float
        Power [W]
    """
    return tf.pow(10, (dbm - 30) / 10)

def matmul_3xN(p, v):
    # squeeze trailing singleton -> (3, 3)
    P = dr.reshape(dr.cuda.ad.TensorXf, p, (3, 3))
    V = v  # (3, N)
    N = V.shape[1]

    # preallocate output (3, N)
    out = dr.empty(dr.cuda.ad.TensorXf, (3, N))

    # compute each output row and assign via slice
    r0 = P[0, 0] * V[0, :] + P[0, 1] * V[1, :] + P[0, 2] * V[2, :]
    r1 = P[1, 0] * V[0, :] + P[1, 1] * V[1, :] + P[1, 2] * V[2, :]
    r2 = P[2, 0] * V[0, :] + P[2, 1] * V[1, :] + P[2, 2] * V[2, :]

    out[0, :] = r0
    out[1, :] = r1
    out[2, :] = r2

    # scale by wavenumber (broadcast is fine)
    return out