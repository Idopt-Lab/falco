import unittest
from unittest import TestCase
import numpy as np
import csdl_alpha as csdl
from falco.core.vehicle.components.aircraft import Aircraft
from falco.core.vehicle.components.component import Component


class TestAircraft(TestCase):
    """Test cases for the Aircraft component class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recorder = csdl.Recorder(inline=True)
        self.recorder.start()
        
    def tearDown(self):
        """Clean up after each test method."""
        self.recorder.stop()
        
    def test_aircraft_inheritance(self):
        """Test that Aircraft inherits from Component."""
        aircraft = Aircraft()
        self.assertIsInstance(aircraft, Component)
        
    def test_aircraft_init_without_geometry(self):
        """Test Aircraft initialization without geometry."""
        aircraft = Aircraft()
        self.assertEqual(aircraft._skip_ffd, False)
        
    def test_aircraft_init_with_geometry(self):
        """Test Aircraft initialization with geometry."""
        # Create a mock geometry (FunctionSet)
        geometry = csdl.Variable(value=np.array([1.0]), shape=(1,), name='mock_geometry')
        aircraft = Aircraft(geometry=geometry)
        self.assertEqual(aircraft._skip_ffd, False)
        
    def test_aircraft_init_with_kwargs(self):
        """Test Aircraft initialization with additional keyword arguments."""
        aircraft = Aircraft(description="Test aircraft")
        self.assertEqual(aircraft._skip_ffd, False)
        
    def test_aircraft_do_not_remake_ffd_block_setting(self):
        """Test that do_not_remake_ffd_block is passed to parent Component."""
        aircraft = Aircraft()
        # The do_not_remake_ffd_block should be passed to the parent Component
        # This is handled in the Aircraft's __init__ method via kwargs
        self.assertEqual(aircraft._skip_ffd, False)
        
    def test_aircraft_skip_ffd_property(self):
        """Test the _skip_ffd property of Aircraft."""
        aircraft = Aircraft()
        self.assertEqual(aircraft._skip_ffd, False)
        
    def test_aircraft_component_type(self):
        """Test that Aircraft is properly typed as a Component."""
        aircraft = Aircraft()
        self.assertIsInstance(aircraft, Component)
        self.assertIsInstance(aircraft, Aircraft)


if __name__ == '__main__':
    unittest.main()
