from falco.core.dynamics.axis import Axis
import csdl_alpha as csdl
import numpy as np
from falco import ureg
from falco.utils.euler_rotations import build_rotation_matrix
from falco.core.dynamics.vector import Vector


class ForcesMoments:
    """Represents a set of forces and moments acting on a body, expressed in a given axis system.

    Attributes
    ----------
    F : Vector
        The force vector.
    M : Vector
        The moment vector.
    axis : Axis
        The axis in which the forces and moments are expressed.
    """
    def __init__(self, force: Vector, moment: Vector):
        """Initialize a ForcesMoments object.

        Parameters
        ----------
        force : Vector
            The force vector.
        moment : Vector
            The moment vector.

        Raises
        ------
        AssertionError
            If the force and moment are not expressed in the same axis.
        """
        assert force.axis == moment.axis, "F and M must be expressed in the same axis"
        self.F = force
        self.M = moment
        self.axis = force.axis



    def transform_to_axis(self, new_axis):
        """Transform the forces and moments to a different axis system.

        Parameters
        ----------
        new_axis : Axis
            The target axis to transform to.
        translate_flag : bool, optional
            Whether to apply translation (default True).
        rotate_flag : bool, optional
            Whether to apply rotation (default True).
        reverse_flag : bool, optional
            Whether to reverse the transformation (default False).

        Returns
        -------
        ForcesMoments
            A new ForcesMoments object in the target axis.
        """
        # # We have a parent axis (B1) and a child axis B2
        # # 1. The forces and moments are in the B2 frame and we want to transform to the B1 frame
        # if self.axis.reference is not None:
        #     if self.axis.reference.name == new_axis.name:
        #         euler = self.axis.euler_angles_vector
        #         seq = self.axis.sequence
        #         displacement = self.axis.translation

        #         orig_force = self.F.vector
        #         orig_moment = self.M.vector

        #         # First perform rotation
        #         if rotate_flag:
        #             inter_force, inter_moment = self.rotate_to_axis(orig_force, orig_moment, euler, seq, reverse=reverse_flag)
        #         else:
        #             inter_force = orig_force
        #             inter_moment = orig_moment

        #         # Then perform displacement
        #         if translate_flag:
        #             new_force, new_moment = self.translate_to_axis(inter_force, inter_moment, displacement)
        #         else:
        #             new_force = inter_force
        #             new_moment = inter_moment
        # # 2. The forces and moments are in the B1 frame and we want to transform to the B2 frame
        # # if it has a name
        # if new_axis.reference is not None:
        #     if new_axis.reference.name == self.axis.name:
        #         euler = new_axis.euler_angles_vector
        #         seq = new_axis.sequence
        #         displacement = new_axis.translation

        #         orig_force = self.F.vector
        #         orig_moment = self.M.vector

        #         # First perform rotation
        #         if rotate_flag:
        #             inter_force, inter_moment = self.rotate_to_axis(orig_force, orig_moment, euler, seq,
        #                                                             reverse=True)
        #         else:
        #             inter_force = orig_force
        #             inter_moment = orig_moment
        #             # Then perform displacement
        #         if translate_flag:
        #             new_force, new_moment = self.translate_to_axis(inter_force, inter_moment, displacement,
        #                                                            reverse=True)
        #         else:
        #             new_force = inter_force
        #             new_moment = inter_moment

        # new_load = ForcesMoments(force=Vector(vector=new_force, axis=new_axis),
        #                          moment=Vector(vector=new_moment, axis=new_axis))
        # return new_load
        
        euler = self.axis.euler_angles_vector - new_axis.euler_angles_vector
        seq = new_axis.sequence

        T = self.axis.translation_from_origin_vector - new_axis.translation_from_origin_vector

        orig_force = self.F.vector
        orig_moment = self.M.vector

        R = build_rotation_matrix(euler, seq)

        new_force = csdl.matvec(R, orig_force)
        new_force.add_tag(orig_force.tags[0])
        new_moment = csdl.matvec(R, orig_moment) + csdl.cross(T, new_force)
        new_moment.add_tag(orig_moment.tags[0])

        return ForcesMoments(force=Vector(vector=new_force, axis=new_axis),
                             moment=Vector(vector=new_moment, axis=new_axis))
    
    @staticmethod
    def rotate_to_axis(F, M, euler_angles, seq, reverse=False):
        """Rotate the force and moment vectors using Euler angles.

        Parameters
        ----------
        F : csdl.Variable
            Force vector.
        M : csdl.Variable
            Moment vector.
        euler_angles : array-like
            Euler angles for rotation.
        seq : str
            Sequence of Euler rotations.
        reverse : bool, optional
            Whether to reverse the rotation (default False).

        Returns
        -------
        tuple
            Rotated force and moment vectors.
        """
        R = build_rotation_matrix(euler_angles, seq)
        if reverse:
            R = csdl.transpose(R)
        F_rot = csdl.matvec(R, F)
        F_rot.add_tag(F.tags[0])
        M_rot = csdl.matvec(R, M)
        M_rot.add_tag(M.tags[0])
        return F_rot, M_rot

    @staticmethod
    def translate_to_axis(F, M, r_vector, reverse=False):
        """Translate the moment vector to a new reference point.

        Parameters
        ----------
        F : csdl.Variable
            Force vector.
        M : csdl.Variable
            Moment vector.
        r_vector : array-like
            Displacement vector for translation.
        reverse : bool, optional
            Whether to reverse the translation (default False).

        Returns
        -------
        tuple
            Force vector (unchanged) and translated moment vector.
        """
        if reverse:
            r_vector = -r_vector
        M_trans = M + csdl.cross(r_vector, F)
        M_trans.add_tag(M.tags[0])
        return F, M_trans

