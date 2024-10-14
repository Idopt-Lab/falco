from flight_simulator.utils.axis import Axis
import csdl_alpha as csdl
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.utils.euler_rotations import build_rotation_matrix


class Vector:
    def __init__(self, vector, axis):
        """
        :param vector: Stores 3 components of the vector in SI units
        :param axis: coordinate system in which the vector is stored
        """

        if isinstance(vector, ureg.Quantity):
            self.vector = csdl.Variable(
                shape=(3,),
            )
            vector_si = vector.to_base_units()
            self.vector.set_value(vector_si.magnitude)
            self.vector.add_tag(str(vector_si.units))
        elif isinstance(vector, csdl.Variable):
            self.vector = vector
        else:
            raise IOError

        assert isinstance(axis, Axis)
        self.axis = axis

    @property
    def magnitude(self):
        return csdl.norm(self.vector)

    def __str__(self):
        print_string = """Vector: %s \nUnit: %s \nAxis: %s""" % \
                       (np.array_str(np.around(self.vector.value, 2)),
                        self.vector.tags[0],
                        self.axis.name)
        return print_string


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

        # There are 2 possibilities
        # 1. The forces and moments are in the B1 frame and we want to transform to the B2 frame
        if parent_or_child_axis.reference.name == self.axis.name:
            euler = parent_or_child_axis.angles
            seq = parent_or_child_axis.sequence
            raise NotImplementedError
        # 2. The forces and moments are in the B2 frame and we want to transform to the B1 frame
        elif self.axis.reference.name == parent_or_child_axis.name:
            euler = self.axis.angles
            seq = self.axis.sequence
            displacement = self.axis.translation
            R = build_rotation_matrix(euler, seq)
            # R = R.transpose()
            # perform vector rotation first
            InterForce = csdl.matvec(R, orig_force)
            InterMoment = csdl.matvec(R, orig_moment)
            # then perform displacement
            newForce = InterForce
            newMoment = InterMoment + csdl.cross(displacement, InterForce)
        else:
            raise IOError

        new_load = ForcesMoments(force=Vector(vector=newForce, axis=parent_or_child_axis),
                                 moment=Vector(vector=newMoment, axis=parent_or_child_axis))
        return new_load


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        translation=np.array([0, 0, 0]) * ureg.meter,
        angles=np.array([0, 0, 0]) * ureg.degree,
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