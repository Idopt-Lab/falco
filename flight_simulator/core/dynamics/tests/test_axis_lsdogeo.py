import unittest
import numpy as np
from flight_simulator import ureg
import lsdo_geo
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo

class TestAxisLsdoGeo(unittest.TestCase):
    def setUp(self):
        # Create a mock geometry object with an evaluate method
        class MockGeometry:
            def evaluate(self, parametric_coords):
                return np.array([1.0, 2.0, 3.0])
        
        self.geometry = MockGeometry()
        self.parametric_coords = [0.5, 0.5, 0.5]
        self.origin = 'test_origin'
        self.axis = AxisLsdoGeo(
            name="test_axis",
            parametric_coords=self.parametric_coords,
            geometry=self.geometry,
            origin="origin",
            phi=np.array([0.1]) * ureg.radian,
            theta=np.array([0.2]) * ureg.radian,
            psi=np.array([0.3]) * ureg.radian
        )

    def test_initialization(self):
        self.assertEqual(self.axis.name, "test_axis")
        self.assertEqual(self.axis.x, 1.0)
        self.assertEqual(self.axis.y, 2.0)
        self.assertEqual(self.axis.z, 3.0)
        self.assertEqual(self.axis.phi.magnitude, 0.1)
        self.assertEqual(self.axis.theta.magnitude, 0.2)
        self.assertEqual(self.axis.psi.magnitude, 0.3)
        self.assertEqual(self.axis.origin, "origin")

    def test_translation_property(self):
        translation = self.axis.translation
        np.testing.assert_array_equal(translation, np.array([1.0, 2.0, 3.0]))

    def test_translation_setter(self):
        # Test that the setter does not change the translation
        self.axis.translation = np.array([4.0, 5.0, 6.0])
        translation = self.axis.translation
        np.testing.assert_array_equal(translation, np.array([1.0, 2.0, 3.0]))

if __name__ == '__main__':
    unittest.main()