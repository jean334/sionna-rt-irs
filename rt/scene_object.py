#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Class representing objects in the scene"""

import drjit as dr
import mitsuba as mi
import os

from .utils import theta_phi_from_unit_vec, rotation_matrix
from .radio_materials import RadioMaterialBase, HolderMaterial
from . import scene as scene_module
from sionna.rt import RadioDevice


class SceneObject:
    # pylint: disable=line-too-long
    r"""
    Class implementing a scene object

    Scene objects can be either created from an existing Mitsuba shape
    or by loading a mesh from a file. In the latter case, a name
    and radio material for the scene object must be provided.

    :param mi_shape: Mitsuba shape.
        Must be provided if ``fname`` is :py:class:`None`.

    :param name: Object name.
        Must be provided if ``fname`` is not :py:class:`None`.

    :param fname: Filename of a valid mesh ( "*.ply" | "*.obj").
        Must be provided if ``mi_shape`` is :py:class:`None`.

    :param radio_material: Radio material of the object.
        Must be provided if ``fname`` is not :py:class:`None`.
    """
    # Counter to handle objects with no name
    NO_NAME_COUNTER = 0

    def __init__(self,
                 mi_shape: mi.Mesh | None=None,
                 name: str | None=None,
                 fname: str | None=None,
                 radio_material: RadioMaterialBase | None=None):

        if mi_shape:
            if not isinstance(mi_shape, mi.Mesh):
                raise ValueError("`mi_shape` must a Mitsuba Shape object")
            if not isinstance(mi_shape.bsdf(), HolderMaterial):
                raise ValueError("The BSDF of `mi_shape` must be a"
                                 " HolderMaterial object")
        elif fname:
            # Mesh type
            mesh_type = os.path.splitext(fname)[1][1:]
            if mesh_type not in ('ply', 'obj'):
                raise ValueError("Invalid mesh type."
                                 " Supported types: `ply` and `obj`")
            if not isinstance(name, str):
                raise ValueError("The `name` of the object to instantiate must"
                                 " be an str")
            if not isinstance(radio_material, RadioMaterialBase):
                raise ValueError("The `radio_material` for the object to"
                                 " instantiate must be a RadioMaterialBase")

            mi_shape = mi.load_dict({'type': mesh_type,
                                     'id' : name,
                                     'filename': fname,
                                     'flip_normals': True,
                                     'bsdf' : {'type': 'holder-material',
                                               'id': f'mat-holder-{name}'}
                                     })
            mi_shape.bsdf().radio_material = radio_material
        else:
            raise ValueError("Either a Mitsuba Shape (mi_shape) or a filename"
                             " (fname) must be provided")

        # Set the Mitsuba shape
        self._mi_shape = mi_shape

        # Scene object to which the object belongs
        self._scene = None

        # Read the ID from the Mitsuba Shape
        # The object ID is the corresponding Mitsuba shape pointer
        # reinterpreted as an UInt32
        self._object_id = dr.reinterpret_array(mi.UInt32,
                                               mi.ShapePtr(mi_shape))[0]

        if mi_shape.id() == "":
            SceneObject.NO_NAME_COUNTER += 1
            name = f"no-name-{SceneObject.NO_NAME_COUNTER}"
            mi_shape.set_id(name)

        # Increment the material counter of objects
        self.radio_material.add_object()

        # Set initial position and orientation of the object
        self._position = mi.Point3f(0,0,0)
        self._orientation = mi.Point3f(0,0,0)
        self._scaling = mi.Float(1.)

    @property
    def scene(self):
        """
        Get/set the scene to which the object belongs. Note that the scene can
        only be set once.

        :type: :py:class:`sionna.rt.Scene`
        """
        return self._scene

    @scene.setter
    def scene(self, scene: scene_module):
        if not isinstance(scene, scene_module.Scene):
            raise ValueError("`scene` must be an instance of Scene")
        if (self._scene is not None) and (self._scene is not scene):
            msg = f"Radio material ('{self.name}') already used by another "\
                "scene"
            raise ValueError(msg)
        self._scene = scene

    @staticmethod
    def shape_id_to_name(shape_id):
        name = shape_id
        if shape_id.startswith("mesh-"):
            name = shape_id[5:]
        return name

    @property
    def name(self):
        r"""Name

        :type: :py:class:`str`
        """
        return SceneObject.shape_id_to_name(self._mi_shape.id())

    @property
    def object_id(self):
        r"""Identifier

        :type: :py:class:`int`
        """
        return self._object_id

    @property
    def mi_shape(self):
        r"""Get/set the Mitsuba shape

        :type: :py:class:`mi.Mesh`
        """
        return self._mi_shape

    @mi_shape.setter
    def mi_shape(self, v: mi.Mesh):
        self._mi_shape = v

    @property
    def radio_material(self):
        r"""Get/set the radio material of the object. Setting can be done by
        using either an instance of :class:`~sionna.rt.RadioMaterialBase` or the
        material name (:py:class:`str`).

        :type: :class:`~sionna.rt.RadioMaterialBase`
        """
        return self._mi_shape.bsdf().radio_material

    @radio_material.setter
    def radio_material(self, mat: HolderMaterial | str | RadioMaterialBase):

        if isinstance(mat, HolderMaterial):
            mat = mat.bsdf

        if isinstance(mat, str) and (self._scene is not None):
            mat_obj = self._scene.get(mat)
            if ( (mat_obj is None) or
                 (not isinstance(mat_obj, RadioMaterialBase)) ):
                err_msg = f"Unknown radio material '{mat}'"
                raise TypeError(err_msg)

        elif not isinstance(mat, RadioMaterialBase):
            err_msg = ("The material must be a material name (str) or an "
                        "instance of RadioMaterialBase")
            raise TypeError(err_msg)

        else:
            mat_obj = mat

        # Current radio material
        current_mat = self.radio_material

        # Add the radio material to the scene
        if self._scene is not None:
            self._scene.add(mat_obj)
            # Ensure that the object and the material belong to the same scene
            if self._scene != mat_obj.scene:
                msg = "Radio material and object are not part of the same scene"
                raise ValueError(msg)

        # Increment the material counter of objects
        mat_obj.add_object()

        # Remove the object from the set of the currently used material, if any
        if isinstance(current_mat, RadioMaterialBase):
            current_mat.remove_object()

        # Effectively update the radio material of the Mitsuba shape
        # via our HolderBSDF proxy object:
        self._mi_shape.bsdf().radio_material = mat_obj

    @property
    def velocity(self):
        r"""Get/set the velocity vector [m/s]

        :type: :py:class:`mi.Vector3f`
        """
        return self._mi_shape.bsdf().velocity

    @velocity.setter
    def velocity(self, v: mi.Vector3f):
        self._mi_shape.bsdf().velocity = v

    @property
    def position(self):
        r"""Get/set the position vector [m] of the center of the object. The
        center is defined as the object's axis-aligned bounding box (AABB).

        :type: :py:class:`mi.Point3f`
        """
        # Bounding box
        bbox_min = self._mi_shape.bbox().min
        bbox_max = self._mi_shape.bbox().max
        position = (bbox_min + bbox_max)*0.5
        return mi.Point3f(position)

    @position.setter
    def position(self, new_position: mi.Point3f):

        # Scene parameters
        scene_params = self._scene.mi_scene_params

        # Use the shape id, and not the object name, to access the Mitsuba
        # scene
        vp_key = self._mi_shape.id() + ".vertex_positions"

        current_vertices = dr.unravel(mi.Point3f, scene_params[vp_key])
        translation_vector = new_position - self.position
        translated_vertices = current_vertices + translation_vector
        scene_params[vp_key] = dr.ravel(translated_vertices)

        scene_params.update()
        self._scene.scene_geometry_updated()

    @property
    def orientation(self):
        r"""Get/set the orientation [rad] specified through three angles
        :math:`(\alpha, \beta, \gamma)` corresponding to a 3D rotation as
        defined in :eq:`rotation`

        :type: :py:class:`mi.Point3f`
        """
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation: mi.Point3f):

        new_orientation = mi.Point3f(new_orientation)

        # Build the transformtation corresponding to the new rotation
        new_rotation = rotation_matrix(new_orientation)

        # Invert the current orientation
        cur_rotation = rotation_matrix(self.orientation)
        inv_cur_rotation = cur_rotation.T

        # Scene parameters
        scene_params = self._scene.mi_scene_params

        # Use the shape id, and not the object name, to access the Mitsuba
        # scene
        vp_key = self._mi_shape.id() + ".vertex_positions"

        # To rotate the object, we need to:
        # 1. Position such that its center is (0,0,0)
        # 2. Undo the current orientation (if any)
        # 3. Apply the new orientation
        # 4. Reposition the object to its current position

        current_vertices = dr.unravel(mi.Point3f, scene_params[vp_key])
        position = self.position
        rotated_vertices = current_vertices - position
        rotated_vertices = new_rotation@inv_cur_rotation@rotated_vertices
        rotated_vertices = rotated_vertices + position
        scene_params[vp_key] = dr.ravel(rotated_vertices)
        scene_params.update()

        self._orientation = new_orientation
        self._scene.scene_geometry_updated()

    @property
    def scaling(self):
        r"""Get/set the scaling

        :type: :py:class:`mi.Float`
        """
        return self._scaling

    @scaling.setter
    def scaling(self, new_scaling: mi.Float):

        if new_scaling <= 0.:
            raise ValueError("Scaling must be positive")

        # Scene parameters
        scene_params = self._scene.mi_scene_params

        # Use the shape id, and not the object name, to access the Mitsuba
        # scene
        vp_key = self._mi_shape.id() + ".vertex_positions"

        current_vertices = dr.unravel(mi.Point3f, scene_params[vp_key])
        current_vertices -= self.position
        scaled_vertices = new_scaling*current_vertices/self._scaling
        scaled_vertices += self.position
        scene_params[vp_key] = dr.ravel(scaled_vertices)

        self._scaling = new_scaling

        scene_params.update()
        self._scene.scene_geometry_updated()

    def look_at(self, target: mi.Point3f | RadioDevice | str):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward an position

        :param target:  A position or the name or instance of an
            object in the scene to point toward to
        """

        # Get position to look at
        if isinstance(target, (SceneObject, RadioDevice)):
            target = target.position
        elif isinstance(target, mi.Point3f):
            pass # Nothing to do
        else:
            raise ValueError("Invalid type for `target`")

        # Compute angles relative to LCS
        x = target - self.position
        x = dr.normalize(x)
        theta, phi = theta_phi_from_unit_vec(x)
        alpha = phi # Rotation around z-axis
        beta = theta - dr.pi/2. # Rotation around y-axis
        gamma = 0.0 # Rotation around x-axis
        self.orientation = mi.Point3f(alpha, beta, gamma)
