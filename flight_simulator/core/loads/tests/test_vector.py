from unittest import TestCase

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments


class TestVector(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        # Setup a mock axis for testing
        self.axis = Axis(
            name='Inertial Axis',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value,
        )

    def test_vector_initialization_with_quantity(self):
        # Create a Quantity vector with units
        vector_quantity = Q_([10, 20, 30], 'newton')
        vector = Vector(vector_quantity, self.axis)

        # Check if vector was initialized with correct values and units
        self.assertEqual(vector.vector.shape, (3,))
        np.testing.assert_array_equal(vector.vector.value, [10, 20, 30])
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')
        self.assertEqual(vector.axis.name, "Inertial Axis")

    def test_vector_initialization_with_nonSI_quantity(self):
        # Create a Quantity vector with units
        vector_quantity = Q_([0, 100, 0], 'lbf')
        vector = Vector(vector_quantity, self.axis)

        # Check if vector was initialized with correct values and units
        self.assertEqual(vector.vector.shape, (3,))
        np.testing.assert_almost_equal(vector.vector.value, [0, 100*4.44822162, 0], decimal=5)
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')
        self.assertEqual(vector.axis.name, "Inertial Axis")

    def test_vector_initialization_with_variable(self):
        # Create a csdl.Variable with a 'newton' tag
        vector_variable = csdl.Variable(shape=(3,), value=np.array([10, 20, 30]))
        vector_variable.add_tag('kilogram * meter / second ** 2')
        vector = Vector(vector_variable, self.axis)

        # Check if vector was initialized with correct values and tags
        np.testing.assert_array_equal(vector.vector.value, [10, 20, 30])
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')

    def test_vector_magnitude(self):
        vector_quantity = Q_([3, 4, 0], 'newton')
        vector = Vector(vector_quantity, self.axis)
        np.testing.assert_array_equal(vector.magnitude.value, np.array([5, ]))
    
    def test_vector_units(self):
        vector_quantity = Q_([10, 20, 30], 'newton')
        vector = Vector(vector_quantity, self.axis)
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')

    def test_vector_axis(self):
        vector_quantity = Q_([10, 20, 30], 'newton')
        vector = Vector(vector_quantity, self.axis)
        self.assertEqual(vector.axis.name, "Inertial Axis")