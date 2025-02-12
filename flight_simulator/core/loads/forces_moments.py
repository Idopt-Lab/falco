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

    def rotate_to_axis(self, parent_or_child_axis):
        """
        :param parent_or_child_axis:
        :return:
        """

        orig_force = self.F.vector
        orig_moment = self.M.vector

        # The loads are in the child axis and we want to transform into the parent axis
        if self.axis.reference.name == parent_or_child_axis.name:
            euler = self.axis.euler_angles
            seq = self.axis.sequence
            displacement = self.axis.translation
            R = build_rotation_matrix(euler, seq)
            # perform vector rotation first
            InterForce = csdl.matvec(R, orig_force)
            InterMoment = csdl.matvec(R, orig_moment)
            # then perform displacement
            newForce = InterForce
            newForce.add_tag(orig_force.tags[0])
            newMoment = InterMoment + csdl.cross(displacement, InterForce)
            newMoment.add_tag(orig_moment.tags[0])
        else:
            print(f"Converting Forces/Moments from in child axis '{self.axis.name}' to parent axis '{parent_or_child_axis.name}'")
            euler_child_to_parent = parent_or_child_axis.euler_angles_vector - self.axis.euler_angles_vector
            seq = parent_or_child_axis.sequence
            translation_child_to_parent = self.axis.translation - parent_or_child_axis.translation
            R_child_to_parent = build_rotation_matrix(euler_child_to_parent, seq)
            InterForce = csdl.matvec(R_child_to_parent, orig_force)
            InterMoment = csdl.matvec(R_child_to_parent, orig_moment)
            newForce = InterForce
            newMoment = InterMoment + csdl.cross(translation_child_to_parent, InterForce)
            if orig_force.tags:
                newForce.add_tag(orig_force.tags[0])
                newMoment.add_tag(orig_moment.tags[0])

            # raise NotImplementedError

        new_load = ForcesMoments(force=Vector(vector=newForce, axis=parent_or_child_axis),
                                 moment=Vector(vector=newMoment, axis=parent_or_child_axis))
        return new_load


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