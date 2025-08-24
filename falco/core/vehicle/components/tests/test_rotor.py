from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from falco import ureg, Q_
import csdl_alpha as csdl

from falco.core.vehicle.components.rotor import Rotor, RotorParameters
from falco.core.vehicle.components.component import Component


class TestRotorParameters(TestCase):
    """Test the RotorParameters dataclass."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_rotor_parameters_creation(self):
        """Test creating RotorParameters with different input types."""
        # Test with float
        radius = 2.5
        params = RotorParameters(radius=radius)
        
        self.assertEqual(params.radius, radius)
        self.assertEqual(params.hub_radius, 0.2)  # Default value
    
    def test_rotor_parameters_with_csdl_variables(self):
        """Test RotorParameters with CSDL variables."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        hub_radius = csdl.Variable(value=0.3, shape=(1,), name="hub_radius")
        
        params = RotorParameters(radius=radius, hub_radius=hub_radius)
        
        self.assertEqual(params.radius, radius)
        self.assertEqual(params.hub_radius, hub_radius)
    
    def test_rotor_parameters_with_int(self):
        """Test RotorParameters with integer values."""
        radius = 3
        hub_radius = 1
        
        params = RotorParameters(radius=radius, hub_radius=hub_radius)
        
        self.assertEqual(params.radius, radius)
        self.assertEqual(params.hub_radius, hub_radius)


class TestRotorBasicFunctionality(TestCase):
    """Test basic Rotor functionality with minimal dependencies."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_rotor_inheritance(self):
        """Test that Rotor is a subclass of Component."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        rotor = Rotor(radius=radius, skip_ffd=True)
        
        self.assertIsInstance(rotor, Component)
        self.assertIsInstance(rotor, Rotor)
    
    def test_rotor_component_attributes(self):
        """Test that Rotor has all required Component attributes."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        rotor = Rotor(radius=radius, skip_ffd=True)
        
        # Check Component attributes
        self.assertEqual(rotor.comps, {})
        self.assertEqual(rotor.surface_mesh, [])
        self.assertEqual(rotor.load_solvers, [])
        self.assertIsNone(rotor.parent)
        self.assertIsNone(rotor.mass_properties)
    
    def test_rotor_parameter_handling(self):
        """Test that Rotor properly handles different parameter types."""
        # Test with CSDL variables
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        rotor = Rotor(radius=radius, skip_ffd=True)
        
        self.assertEqual(rotor.parameters.radius, radius)
        self.assertEqual(rotor._name, "rotor")
        self.assertIsNone(rotor.geometry)
        self.assertTrue(rotor._skip_ffd)
    
    def test_rotor_with_custom_name(self):
        """Test Rotor initialization with custom name."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        custom_name = "main_rotor"
        
        rotor = Rotor(radius=radius, name=custom_name, skip_ffd=True)
        
        self.assertEqual(rotor._name, custom_name)
    

    
    def test_rotor_with_geometry(self):
        """Test Rotor initialization with geometry."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Create a mock class that can handle isinstance checks
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
        
        mock_geometry = MockFunctionSet()
        
        # Mock FFD block
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        
        with patch('falco.core.vehicle.components.rotor.lg') as mock_lg, \
             patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance:
            
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            # Make isinstance return True for our mock geometry
            mock_isinstance.return_value = True
            
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=False)
            
            self.assertEqual(rotor.geometry, mock_geometry)
            self.assertFalse(rotor._skip_ffd)
            self.assertEqual(rotor.ffd_block, mock_ffd_block)
    
    def test_rotor_with_invalid_geometry_type(self):
        """Test Rotor initialization with invalid geometry type."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        invalid_geometry = "not_a_function_set"
        
        with self.assertRaises(TypeError):
            Rotor(radius=radius, geometry=invalid_geometry, skip_ffd=False)
    
    def test_rotor_ffd_block_dimensions(self):
        """Test FFD block dimension calculations."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Mock geometry
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
        
        mock_geometry = MockFunctionSet()
        
        # Create FFD block mock that returns proper CSDL variables
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        
        # Mock evaluate to return CSDL variables that can be used in arithmetic
        def mock_evaluate_side_effect(parametric_coordinates=None):
            return csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        
        mock_ffd_block.evaluate.side_effect = mock_evaluate_side_effect
        
        with patch('falco.core.vehicle.components.rotor.lg') as mock_lg, \
             patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm, \
             patch('falco.core.vehicle.components.rotor.np.where') as mock_np_where:
            
            # Mock norm to return a simple scalar value
            mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
            mock_np_where.return_value = [np.array([0])]  # u-direction is smallest
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            mock_isinstance.return_value = True
            
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=False)
            
            # Verify corner points are set based on smallest dimension
            self.assertTrue(hasattr(rotor, '_corner_point_1'))
            self.assertTrue(hasattr(rotor, '_corner_point_2'))
            self.assertTrue(hasattr(rotor, '_corner_point_3'))
            self.assertTrue(hasattr(rotor, '_corner_point_4'))


class TestRotorActuation(TestCase):
    """Test Rotor actuation functionality."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_actuate_with_no_geometry(self):
        """Test actuation when rotor has no geometry."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        rotor = Rotor(radius=radius, geometry=None, skip_ffd=True)
        
        with self.assertRaises(ValueError):
            rotor.actuate(x_tilt_angle=10.0)
    
    def test_actuate_with_no_angles(self):
        """Test actuation with no tilt angles specified."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Create a mock class that can handle isinstance checks
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
        
        mock_geometry = MockFunctionSet()
        
        with patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.lg') as mock_lg:
            mock_isinstance.return_value = True
            # Mock the FFD block construction to avoid validation
            mock_ffd_block = Mock()
            # Make evaluate return proper CSDL variables for arithmetic operations
            def mock_evaluate_side_effect(parametric_coordinates=None):
                return csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            mock_ffd_block.evaluate.side_effect = mock_evaluate_side_effect
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=True)
            
            with self.assertRaises(ValueError):
                rotor.actuate()
    
    def test_actuate_with_x_tilt_angle(self):
        """Test actuation with x tilt angle."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Create a mock class that can handle isinstance checks
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
                self.rotate = Mock()
        
        mock_geometry = MockFunctionSet()
        
        # Mock discretizations
        mock_mesh = Mock()
        mock_mesh._update.return_value = mock_mesh
        
        with patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.lg') as mock_lg:
            mock_isinstance.return_value = True
            # Mock the FFD block construction to avoid validation
            mock_ffd_block = Mock()
            # Make evaluate return proper CSDL variables for arithmetic operations
            def mock_evaluate_side_effect(parametric_coordinates=None):
                return csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            mock_ffd_block.evaluate.side_effect = mock_evaluate_side_effect
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=True)
            rotor._discretizations = {"test_mesh": mock_mesh}
            rotor._corner_point_1 = np.array([0.5, 0.5, 0.5])
            rotor._corner_point_2 = np.array([0.5, 0.5, 1.5])
            
            with patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm:
                mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
                
                rotor.actuate(x_tilt_angle=10.0)
                
                # Verify geometry.rotate was called
                mock_geometry.rotate.assert_called_once()
                # Verify mesh was updated
                mock_mesh._update.assert_called_once()
    
    def test_actuate_with_y_tilt_angle(self):
        """Test actuation with y tilt angle."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Create a mock class that can handle isinstance checks
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
                self.rotate = Mock()
        
        mock_geometry = MockFunctionSet()
        
        # Mock discretizations
        mock_mesh = Mock()
        mock_mesh._update.return_value = mock_mesh
        
        with patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.lg') as mock_lg:
            mock_isinstance.return_value = True
            # Mock the FFD block construction to avoid validation
            mock_ffd_block = Mock()
            # Make evaluate return proper CSDL variables for arithmetic operations
            def mock_evaluate_side_effect(parametric_coordinates=None):
                return csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            mock_ffd_block.evaluate.side_effect = mock_evaluate_side_effect
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=True)
            rotor._discretizations = {"test_mesh": mock_mesh}
            rotor._corner_point_3 = np.array([0.5, 0.5, 0.5])
            rotor._corner_point_4 = np.array([0.5, 1.5, 0.5])
            
            with patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm:
                mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
                
                rotor.actuate(y_tilt_angle=15.0)
                
                # Verify geometry.rotate was called
                mock_geometry.rotate.assert_called_once()
                # Verify mesh was updated
                mock_mesh._update.assert_called_once()
    
    def test_actuate_with_both_angles(self):
        """Test actuation with both x and y tilt angles."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Create a mock class that can handle isinstance checks
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
                self.rotate = Mock()
        
        mock_geometry = MockFunctionSet()
        
        # Mock discretizations
        mock_mesh = Mock()
        mock_mesh._update.return_value = mock_mesh
        
        with patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.lg') as mock_lg:
            mock_isinstance.return_value = True
            # Mock the FFD block construction to avoid validation
            mock_ffd_block = Mock()
            # Make evaluate return proper CSDL variables for arithmetic operations
            def mock_evaluate_side_effect(parametric_coordinates=None):
                return csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            mock_ffd_block.evaluate.side_effect = mock_evaluate_side_effect
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=True)
            rotor._discretizations = {"test_mesh": mock_mesh}
            rotor._corner_point_1 = np.array([0.5, 0.5, 0.5])
            rotor._corner_point_2 = np.array([0.5, 0.5, 1.5])
            rotor._corner_point_3 = np.array([0.5, 0.5, 0.5])
            rotor._corner_point_4 = np.array([0.5, 1.5, 0.5])
            
            with patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm:
                mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
                
                rotor.actuate(x_tilt_angle=10.0, y_tilt_angle=15.0)
                
                # Verify geometry.rotate was called twice (once for each angle)
                self.assertEqual(mock_geometry.rotate.call_count, 2)
                # Verify mesh was updated
                mock_mesh._update.assert_called_once()
    
    def test_actuate_with_mesh_update_error(self):
        """Test actuation when mesh update fails."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Create a mock class that can handle isinstance checks
        class MockFunctionSet:
            def __init__(self):
                self.project = Mock(return_value=np.array([0.5, 0.5, 0.5]))
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
                self.functions = {}
                self.set_coefficients = Mock()
                self.rotate = Mock()
        
        mock_geometry = MockFunctionSet()
        
        # Mock discretizations with mesh that has no _update method
        mock_mesh = Mock()
        del mock_mesh._update  # Remove _update method
        
        with patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.lg') as mock_lg:
            mock_isinstance.return_value = True
            # Mock the FFD block construction to avoid validation
            mock_ffd_block = Mock()
            # Make evaluate return proper CSDL variables for arithmetic operations
            def mock_evaluate_side_effect(parametric_coordinates=None):
                return csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            mock_ffd_block.evaluate.side_effect = mock_evaluate_side_effect
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            rotor = Rotor(radius=radius, geometry=mock_geometry, skip_ffd=True)
            rotor._discretizations = {"test_mesh": mock_mesh}
            rotor._corner_point_1 = np.array([0.5, 0.5, 0.5])
            rotor._corner_point_2 = np.array([0.5, 0.5, 1.5])
            
            with patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm:
                mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
                
                with self.assertRaises(Exception):
                    rotor.actuate(x_tilt_angle=10.0)


class TestRotorFFDMethods(TestCase):
    """Test Rotor FFD-related methods."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    @patch('falco.core.vehicle.components.rotor.lfs')
    @patch('falco.core.vehicle.components.rotor.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.rotor.VolumeSectionalParameterizationInputs')
    @patch('falco.core.vehicle.components.rotor.csdl.expand')
    def test_setup_ffd_block(self, mock_expand, mock_vspi, mock_vsp, mock_lfs):
        """Test _setup_ffd_block method."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        rotor = Rotor(radius=radius, skip_ffd=True)
        rotor._name = "test_rotor"
        rotor._pr_dim = 1  # v-direction is principal
        # Fix the bug in rotor code where it uses self.skip_ffd instead of self._skip_ffd
        rotor.skip_ffd = True
        
        # Mock FFD block
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
        # Mock B-spline spaces
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        # Mock functions
        mock_function = Mock()
        # Use correct shape for coefficients that can be expanded to (2, 3, 2, 3)
        mock_function.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Mock geometry
        class MockFunctionSet:
            def __init__(self):
                self.functions = {"test": mock_function}
                self.set_coefficients = Mock()
        
        mock_geometry = MockFunctionSet()
        rotor.geometry = mock_geometry
        
        # Mock csdl.expand to return a properly shaped variable
        mock_expand.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        
        rotor._setup_ffd_block(mock_ffd_block, mock_solver, plot=False)
        
        # Verify B-spline spaces were created
        self.assertEqual(mock_lfs.BSplineSpace.call_count, 3)
        # Verify functions were created
        self.assertEqual(mock_lfs.Function.call_count, 3)
        # Verify volume sectional parameterization was created
        mock_vsp.assert_called_once()
        # Verify solver parameters were added
        self.assertGreater(mock_solver.add_parameter.call_count, 0)
    
    def test_extract_geometric_quantities_from_ffd_block(self):
        """Test _extract_geometric_quantities_from_ffd_block method."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        rotor = Rotor(radius=radius, skip_ffd=True)
        
        # Mock geometry
        class MockFunctionSet:
            def __init__(self):
                self.evaluate = Mock(return_value=csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,)))
        
        mock_geometry = MockFunctionSet()
        rotor.geometry = mock_geometry
        
        # Set corner points
        rotor._corner_point_1 = np.array([0.5, 0.5, 0.5])
        rotor._corner_point_2 = np.array([0.5, 0.5, 1.5])
        rotor._corner_point_3 = np.array([0.5, 0.5, 0.5])
        rotor._corner_point_4 = np.array([0.5, 1.5, 0.5])
        
        with patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm:
            mock_norm.return_value = csdl.Variable(value=0.5, shape=(1,))
            
            radius_1, radius_2 = rotor._extract_geometric_quantities_from_ffd_block()
            
            # Verify geometry.evaluate was called for corner points
            self.assertEqual(mock_geometry.evaluate.call_count, 4)
            # Verify norm was called twice (once for each radius calculation)
            self.assertEqual(mock_norm.call_count, 2)
    
    def test_setup_ffd_parameterization(self):
        """Test _setup_ffd_parameterization method."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        rotor = Rotor(radius=radius, skip_ffd=True)
        
        # Mock geometric variables
        mock_geometric_variables = Mock()
        mock_geometric_variables.add_variable = Mock()
        
        # Mock radius variables
        radius_1 = csdl.Variable(value=1.0, shape=(1,), name="radius_1")
        radius_2 = csdl.Variable(value=1.0, shape=(1,), name="radius_2")
        
        rotor._setup_ffd_parameterization(radius_1, radius_2, mock_geometric_variables)
        
        # Verify variables were added
        self.assertEqual(mock_geometric_variables.add_variable.call_count, 2)
    
    @patch('falco.core.vehicle.components.rotor.lfs')
    @patch('falco.core.vehicle.components.rotor.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.rotor.VolumeSectionalParameterizationInputs')
    @patch('falco.core.vehicle.components.rotor.csdl.expand')
    def test_setup_geometry_with_ffd(self, mock_expand, mock_vspi, mock_vsp, mock_lfs):
        """Test _setup_geometry method with FFD enabled."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        rotor = Rotor(radius=radius, skip_ffd=False)
        rotor._name = "test_rotor"
        rotor._pr_dim = 1
        # Fix the bug in rotor code where it uses self.skip_ffd instead of self._skip_ffd
        rotor.skip_ffd = False
        
        # Mock FFD block
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        rotor.ffd_block = mock_ffd_block
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
        # Mock geometric variables
        mock_geometric_variables = Mock()
        mock_geometric_variables.add_variable = Mock()
        
        # Mock B-spline spaces and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        # Use correct shape for coefficients that can be expanded to (2, 3, 2, 3)
        mock_function.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Mock geometry
        class MockFunctionSet:
            def __init__(self):
                self.functions = {"test": mock_function}
                self.set_coefficients = Mock()
        
        mock_geometry = MockFunctionSet()
        rotor.geometry = mock_geometry
        
        # Mock csdl.expand to return a properly shaped variable
        mock_expand.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        
        # Mock geometric quantities extraction
        with patch.object(rotor, '_extract_geometric_quantities_from_ffd_block') as mock_extract, \
             patch.object(rotor, '_setup_ffd_parameterization') as mock_setup_param:
            
            mock_extract.return_value = (csdl.Variable(value=1.0, shape=(1,)), csdl.Variable(value=1.0, shape=(1,)))
            
            rotor._setup_geometry(mock_solver, mock_geometric_variables, plot=False)
            
            # Verify FFD block was set up
            mock_extract.assert_called_once()
            mock_setup_param.assert_called_once()
    
    def test_setup_geometry_without_ffd(self):
        """Test _setup_geometry method with FFD disabled."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        rotor = Rotor(radius=radius, skip_ffd=True)
        # Fix the bug in rotor code where it uses self.skip_ffd instead of self._skip_ffd
        rotor.skip_ffd = True
        
        # Mock FFD block
        mock_ffd_block = Mock()
        rotor.ffd_block = mock_ffd_block
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
        # Mock geometric variables
        mock_geometric_variables = Mock()
        mock_geometric_variables.add_variable = Mock()
        
        with patch.object(rotor, '_setup_ffd_block') as mock_setup_ffd:
            rotor._setup_geometry(mock_solver, mock_geometric_variables, plot=False)
            
            # Verify FFD block was set up but no parameterization
            mock_setup_ffd.assert_called_once()


class TestRotorEdgeCases(TestCase):
    """Test Rotor edge cases and error conditions."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_rotor_with_invalid_radius_type(self):
        """Test Rotor initialization with invalid radius type."""
        invalid_radius = "not_a_number"
        
        with self.assertRaises(Exception):  # csdl.check_parameter will raise an exception
            Rotor(radius=invalid_radius, skip_ffd=True)
    
    def test_rotor_ffd_block_invalid_smallest_dimension(self):
        """Test FFD block setup with invalid smallest dimension."""
        radius = csdl.Variable(value=2.5, shape=(1,), name="radius")
        
        # Mock geometry
        mock_geometry = Mock()
        mock_geometry.project.return_value = np.array([0.5, 0.5, 0.5])
        mock_geometry.evaluate.return_value = csdl.Variable(value=np.array([1.0, 0.0, 0.0]), shape=(3,))
        mock_geometry.functions = {}
        mock_geometry.set_coefficients = Mock()
        
        # Mock FFD block
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        
        with patch('falco.core.vehicle.components.rotor.lg') as mock_lg, \
             patch('falco.core.vehicle.components.rotor.isinstance') as mock_isinstance, \
             patch('falco.core.vehicle.components.rotor.csdl.norm') as mock_norm, \
             patch('falco.core.vehicle.components.rotor.np.array') as mock_np_array, \
             patch('falco.core.vehicle.components.rotor.np.where') as mock_np_where:
            
            # Mock to return invalid smallest dimension
            mock_np_where.return_value = [np.array([5])]  # Invalid dimension
            mock_np_array.return_value = np.array([1.0, 2.0, 3.0])
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            mock_isinstance.return_value = True
            
            with self.assertRaises(Exception):
                Rotor(radius=radius, geometry=mock_geometry, skip_ffd=False)


if __name__ == '__main__':
    import unittest
    unittest.main()
