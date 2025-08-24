import unittest
from unittest import TestCase
import numpy as np
import csdl_alpha as csdl
from falco.core.vehicle.components.powertrain import Powertrain
from falco.core.vehicle.components.component import Component


class TestPowertrain(TestCase):
    """Test cases for the Powertrain component class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recorder = csdl.Recorder(inline=True)
        self.recorder.start()
        
    def tearDown(self):
        """Clean up after each test method."""
        self.recorder.stop()
        
    def test_powertrain_inheritance(self):
        """Test that Powertrain inherits from Component."""
        powertrain = Powertrain()
        self.assertIsInstance(powertrain, Component)
        
    def test_powertrain_init_without_geometry(self):
        """Test Powertrain initialization without geometry."""
        powertrain = Powertrain()
        self.assertEqual(powertrain._skip_ffd, True)
        
    def test_powertrain_init_with_geometry(self):
        """Test Powertrain initialization with geometry."""
        # Create a mock geometry (FunctionSet)
        geometry = csdl.Variable(value=np.array([1.0]), shape=(1,), name='mock_geometry')
        powertrain = Powertrain(geometry=geometry)
        self.assertEqual(powertrain._skip_ffd, True)
        
    def test_powertrain_init_with_kwargs(self):
        """Test Powertrain initialization with additional keyword arguments."""
        powertrain = Powertrain(description="Test powertrain")
        self.assertEqual(powertrain._skip_ffd, True)
        
    def test_powertrain_skip_ffd_property(self):
        """Test the _skip_ffd property of Powertrain."""
        powertrain = Powertrain()
        self.assertEqual(powertrain._skip_ffd, True)
        
    def test_powertrain_component_type(self):
        """Test that Powertrain is properly typed as a Component."""
        powertrain = Powertrain()
        self.assertIsInstance(powertrain, Component)
        self.assertIsInstance(powertrain, Powertrain)
        
    def test_powertrain_geometry_handling(self):
        """Test that Powertrain properly handles geometry parameter."""
        # Test with None geometry
        powertrain_none = Powertrain(geometry=None)
        self.assertEqual(powertrain_none._skip_ffd, True)
        
        # Test with csdl.Variable geometry
        geometry_var = csdl.Variable(value=np.array([1.0, 2.0, 3.0]), shape=(3,), name='geometry')
        powertrain_var = Powertrain(geometry=geometry_var)
        self.assertEqual(powertrain_var._skip_ffd, True)


if __name__ == '__main__':
    unittest.main()
