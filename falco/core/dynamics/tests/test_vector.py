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
    
        # Standard axis
        self.axis = Axis(
            name="TestAxis",
            x=np.array([0,]) * ureg.meter,
            y=np.array([0,]) * ureg.meter,
            z=np.array([1,]) * ureg.meter,
            phi=np.array([0,]) * ureg.radian,
            theta=np.array([0,]) * ureg.radian,
            psi=np.array([0,]) * ureg.radian,                     
            origin=ValidOrigins.Inertial.value
        )

        # Rotated axis for transformation tests
        self.rotated_axis = Axis(
            name="RotatedAxis",
            x=np.array([0,]) * ureg.meter,
            y=np.array([0,]) * ureg.meter,
            z=np.array([1,]) * ureg.meter,
            phi=np.array([np.pi/4,]) * ureg.radian,
            theta=np.array([np.pi/6,]) * ureg.radian,
            psi=np.array([np.pi/3,]) * ureg.radian,
            origin=ValidOrigins.Inertial.value
        )

        class MockGeometry:
            def evaluate(self, parametric_coords):
                return np.array([1.0, 2.0, 3.0]) * ureg.meter
        
        self.geometry = MockGeometry()
        self.parametric_coords = [0.5, 0.5, 0.5]
        self.axis_lsdogeo = AxisLsdoGeo(
            name="TestAxisLsdoGeo",
            origin=ValidOrigins.Inertial.value, 
            parametric_coords=self.parametric_coords, 
            geometry=self.geometry
        )

    def test_vector_initialization(self):
        """Test different ways of initializing vectors"""
        # Test with quantity
        vector_quantity = 3 * ureg.meter
        vector = Vector(vector_quantity, self.axis)
        self.assertIsInstance(vector.vector, csdl.Variable)
        self.assertEqual(vector.vector.shape, (3,))
        self.assertEqual(vector.vector.tags[0], 'meter')
        
        # Test with numpy array
        np_vector = np.array([1, 2, 3])
        vector = Vector(np_vector * ureg.meter, self.axis)
        self.assertEqual(vector.vector.shape, (3,))
        
        # Test with list
        list_vector = [4, 5, 6]
        vector = Vector(np.array(list_vector) * ureg.meter, self.axis)
        self.assertEqual(vector.vector.shape, (3,))

    def test_vector_operations(self):
        """Test vector mathematical operations"""
        v1 = Vector(np.array([1, 0, 0]) * ureg.meter, self.axis)
        v2 = Vector(np.array([0, 1, 0]) * ureg.meter, self.axis)
        
        # Test dot product
        if hasattr(Vector, 'dot'):
            dot_product = v1.dot(v2)
            self.assertAlmostEqual(dot_product.value, 0)
        
        # Test cross product
        if hasattr(Vector, 'cross'):
            cross_product = v1.cross(v2)
            self.assertTrue(np.allclose(
                cross_product.vector.value, 
                np.array([0, 0, 1])
            ))

    def test_vector_magnitude(self):
        """Test vector magnitude calculations"""
        # Zero vector
        zero_vector = Vector(np.zeros(3) * ureg.meter, self.axis)
        self.assertAlmostEqual(zero_vector.magnitude.value, 0)
        
        # Unit vector
        unit_vector = Vector(np.array([1, 0, 0]) * ureg.meter, self.axis)
        self.assertAlmostEqual(unit_vector.magnitude.value, 1)
        
        # General vector
        vector = Vector(np.array([3, 4, 0]) * ureg.meter, self.axis)
        self.assertAlmostEqual(vector.magnitude.value, 5)

    def test_vector_units(self):
        """Test vector unit handling"""
        # Test different units
        vector_meters = Vector(3 * ureg.meter, self.axis)
        vector_feet = Vector(3 * ureg.feet, self.axis)
        
        self.assertEqual(vector_meters.vector.tags[0], 'meter')
        self.assertNotEqual(vector_meters.magnitude.value, 
                          vector_feet.magnitude.value)

    def test_vector_with_csdl_variable(self):
        """Test CSDL variable integration"""
        csdl_vector = csdl.Variable(shape=(3,), value=np.array([3, 3, 3]))
        vector = Vector(csdl_vector, self.axis)
        self.assertIsInstance(vector.vector, csdl.Variable)
        self.assertEqual(vector.vector.shape, (3,))
        self.assertEqual(vector.axis, self.axis)

if __name__ == '__main__':
    unittest.main()