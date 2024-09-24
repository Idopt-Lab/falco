from flight_simulator.utils.axis import Axis
import csdl_alpha as csdl
import numpy as np
from flight_simulator import ureg, Q_


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