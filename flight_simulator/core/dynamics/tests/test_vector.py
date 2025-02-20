import unittest
import csdl_alpha as csdl
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.dynamics.vector import Vector

class TestVectorInitialization(unittest.TestCase):
    def setUp(self):
        self.axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin='Inertial'
        )
        self.axis_lsdogeo = AxisLsdoGeo(
            name='Geo Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin='Geo'
        )

    def test_init_with_quantity(self):
        vector = Q_([1, 2, 3], 'meter')
        vec = Vector(vector, self.axis)
        self.assertIsInstance(vec.vector, csdl.Variable)
        self.assertEqual(vec.vector.shape, (3,))
        self.assertEqual(vec.axis, self.axis)

    def test_init_with_csdl_variable(self):
        vector = csdl.Variable(shape=(3,))
        vector.set_value([1, 2, 3])
        vec = Vector(vector, self.axis)
        self.assertIsInstance(vec.vector, csdl.Variable)
        self.assertEqual(vec.vector.shape, (3,))
        self.assertEqual(vec.axis, self.axis)

    def test_init_with_invalid_vector_type(self):
        with self.assertRaises(IOError):
            Vector([1, 2, 3], self.axis)

    def test_init_with_invalid_axis_type(self):
        vector = Q_([1, 2, 3], 'meter')
        with self.assertRaises(TypeError):
            Vector(vector, "invalid_axis")

if __name__ == '__main__':
    unittest.main()