import unittest
from flight_simulator.core.dynamics.vector import Vector
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator import ureg
import csdl_alpha as csdl
import numpy as np

class TestVector(unittest.TestCase):

    def setUp(self):

        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
        self.axis = Axis(name="TestAxis",origin=ValidOrigins.Inertial.value)

        class MockGeometry:
            def evaluate(self, parametric_coords):
                return np.array([1.0, 2.0, 3.0]) * ureg.meter
        
        self.geometry = MockGeometry()
        self.parametric_coords = [0.5, 0.5, 0.5]

        self.axis_lsdogeo = AxisLsdoGeo(name="TestAxisLsdoGeo",origin=ValidOrigins.Inertial.value, parametric_coords=self.parametric_coords, geometry=self.geometry)

    def test_vector_with_quantity(self):
        vector_quantity = 3 * ureg.meter
        vector = Vector(vector_quantity, self.axis)
        self.assertIsInstance(vector.vector, csdl.Variable)
        self.assertEqual(vector.vector.shape, (3,))
        self.assertEqual(vector.vector.tags[0], 'meter')
        self.assertEqual(vector.axis, self.axis)

    def test_vector_with_csdl_variable(self):
        csdl_vector = csdl.Variable(shape=(3,), value=np.array([3, 3, 3]))
        vector = Vector(csdl_vector, self.axis)
        self.assertIsInstance(vector.vector, csdl.Variable)
        self.assertEqual(vector.vector.shape, (3,))
        self.assertEqual(vector.axis, self.axis)

    def test_invalid_vector_type(self):
        with self.assertRaises(IOError):
            Vector("invalid_vector", self.axis)

    def test_invalid_axis_type(self):
        vector_quantity = 3 * ureg.meter
        with self.assertRaises(TypeError):
            Vector(vector_quantity, "invalid_axis")

    def test_vector_magnitude(self):
        vector_quantity = 3 * ureg.meter
        vector = Vector(vector_quantity, self.axis)
        self.assertAlmostEqual(vector.magnitude.value, np.linalg.norm([3, 3, 3]))

    def test_vector_str(self):
        vector_quantity = 3 * ureg.meter
        vector = Vector(vector_quantity, self.axis)
        expected_str = "Vector: [3. 3. 3.] \nUnit: meter \nAxis: TestAxis"
        self.assertEqual(str(vector), expected_str)

if __name__ == '__main__':
    unittest.main()