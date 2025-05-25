from flight_simulator.core.dynamics.axis import Axis
import csdl_alpha as csdl
import numpy as np
from flight_simulator import ureg
from flight_simulator.utils.euler_rotations import build_rotation_matrix
from flight_simulator.core.dynamics.vector import Vector


class ForcesMoments:
    def __init__(self, force: Vector, moment: Vector):
        assert force.axis == moment.axis, "F and M must be expressed in the same axis"
        self.F = force
        self.M = moment
        self.axis = force.axis



    def transform_to_axis(self, parent_or_child_axis, translate_flag = True, rotate_flag=True, reverse_flag=False):

        # We have a parent axis (B1) and a child axis B2
        # 1. The forces and moments are in the B2 frame and we want to transform to the B1 frame
        if self.axis.reference is not None:
            if self.axis.reference.name == parent_or_child_axis.name:
                euler = self.axis.euler_angles_vector
                seq = self.axis.sequence
                displacement = self.axis.translation

                orig_force = self.F.vector
                orig_moment = self.M.vector

                # First perform rotation
                if rotate_flag:
                    inter_force, inter_moment = self.rotate_to_axis(orig_force, orig_moment, euler, seq, reverse=reverse_flag)
                else:
                    inter_force = orig_force
                    inter_moment = orig_moment

                # Then perform displacement
                if translate_flag:
                    new_force, new_moment = self.translate_to_axis(inter_force, inter_moment, displacement)
                else:
                    new_force = inter_force
                    new_moment = inter_moment
        # 2. The forces and moments are in the B1 frame and we want to transform to the B2 frame
        # if it has a name
        if parent_or_child_axis.reference is not None:
            if parent_or_child_axis.reference.name == self.axis.name:
                euler = parent_or_child_axis.euler_angles_vector
                seq = parent_or_child_axis.sequence
                displacement = parent_or_child_axis.translation

                orig_force = self.F.vector
                orig_moment = self.M.vector

                # First perform rotation
                if rotate_flag:
                    inter_force, inter_moment = self.rotate_to_axis(orig_force, orig_moment, euler, seq,
                                                                    reverse=True)
                else:
                    inter_force = orig_force
                    inter_moment = orig_moment
                    # Then perform displacement
                if translate_flag:
                    new_force, new_moment = self.translate_to_axis(inter_force, inter_moment, displacement,
                                                                   reverse=True)
                else:
                    new_force = inter_force
                    new_moment = inter_moment

        new_load = ForcesMoments(force=Vector(vector=new_force, axis=parent_or_child_axis),
                                 moment=Vector(vector=new_moment, axis=parent_or_child_axis))
        return new_load

    @staticmethod
    def rotate_to_axis(F, M, euler_angles, seq, reverse=False):
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
        if reverse:
            r_vector = -r_vector
        M_trans = M + csdl.cross(r_vector, F)
        M_trans.add_tag(M.tags[0])
        return F, M_trans


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        translation=np.array([0, 0, 0]) * ureg.meter,
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([5, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        origin='inertial'
    )

    # Define as a Pint Quantity
    force_vector_1 = Vector(vector=np.array([0, 400, 0])*ureg.lbf, axis=inertial_axis)
    print(force_vector_1.magnitude.value)
    print(force_vector_1)

    # Define as a CSDL variable
    csdl_vector = csdl.Variable(shape=(3,), value=np.array([0, 400, 0]), tags=[str(ureg.newton)])
    force_vector_2 = Vector(vector=csdl_vector, axis=inertial_axis)
    print(force_vector_2.magnitude.value)
    pass