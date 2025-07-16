from falco.core.dynamics.axis import Axis
from falco.core.dynamics.axis_lsdogeo import AxisLsdoGeo
import csdl_alpha as csdl
from falco import ureg
import numpy as np
from falco.core.dynamics.axis_lsdogeo import AxisLsdoGeo

class Vector:
    """Represents a 3D vector in a specified coordinate axis system.

    Supports initialization from a Pint Quantity (with units) or a CSDL variable.
    Stores the vector in SI units and associates it with an axis.

    Attributes
    ----------
    vector : csdl.Variable
        The 3-component vector in SI units.
    axis : Axis or AxisLsdoGeo
        The coordinate system in which the vector is defined.
    magnitude : csdl.Variable
        The Euclidean norm (magnitude) of the vector.
    """
    def __init__(self, vector, axis):
        """
        Initialize a Vector object.

        Parameters
        ----------
        vector : ureg.Quantity or csdl.Variable
            3-component vector, either as a Pint Quantity (with units) or a CSDL variable.
        axis : Axis or AxisLsdoGeo
            The coordinate system in which the vector is stored.

        Raises
        ------
        IOError
            If the vector is not a recognized type.
        TypeError
            If the axis is not an instance of Axis or AxisLsdoGeo.
        Exception
            If the axis is not assigned correctly.
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
            self.vector.add_tag('csdl_variable')
        else:
            raise IOError

        if not isinstance(axis, (Axis, AxisLsdoGeo)):
            raise TypeError("axis must be an instance of Axis or AxisLSDOGeo")
        self.axis = axis
        self.magnitude = csdl.norm(self.vector)
        # Ensure axis is assigned correctly
        if not hasattr(self, 'axis'):
            raise Exception("Axis not assigned correctly.")


    def __str__(self):
        """Return a string representation of the vector, including values, units, and axis name.

        Returns
        -------
        str
            String representation of the vector.
        """
        print_string = """Vector: %s \nUnit: %s \nAxis: %s""" % \
                       (np.array_str(np.around(self.vector.value, 2)),
                        self.vector.tags[0],
                        self.axis.name)
        return print_string
