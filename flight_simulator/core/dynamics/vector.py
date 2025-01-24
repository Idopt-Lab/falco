from flight_simulator.core.dynamics.axis import Axis
import csdl_alpha as csdl
from flight_simulator import ureg
import numpy as np

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
        self.magnitude = csdl.norm(self.vector)

    def __str__(self):
        print_string = """Vector: %s \nUnit: %s \nAxis: %s""" % \
                       (np.array_str(np.around(self.vector.value, 2)),
                        self.vector.tags[0],
                        self.axis.name)
        return print_string