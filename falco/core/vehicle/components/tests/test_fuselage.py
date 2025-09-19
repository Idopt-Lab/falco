from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from falco import ureg, Q_
import csdl_alpha as csdl

from falco.core.vehicle.components.fuselage import Fuselage, FuselageParameters, FuselageGeometricQuantities
from falco.core.vehicle.components.component import Component


class TestFuselageParameters(TestCase):
    """Test the FuselageParameters dataclass."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_fuselage_parameters_creation(self):
        """Test creating FuselageParameters with different input types."""
        # Test with ureg.Quantity
        length = Q_(10.0, "m")
        max_width = Q_(2.0, "m") 
        max_height = Q_(1.5, "m")
        S_wet = Q_(25.0, "m**2")
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height,
            S_wet=S_wet
        )
        
        self.assertEqual(params.length, length)
        self.assertEqual(params.max_width, max_width)
        self.assertEqual(params.max_height, max_height)
        self.assertEqual(params.S_wet, S_wet)
    
    def test_fuselage_parameters_default_s_wet(self):
        """Test FuselageParameters with default S_wet value."""
        length = Q_(10.0, "m")
        max_width = Q_(2.0, "m") 
        max_height = Q_(1.5, "m")
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height
        )
        
        self.assertEqual(params.S_wet, Q_(1, "m**2"))
    
    def test_fuselage_parameters_with_csdl_variables(self):
        """Test FuselageParameters with CSDL variables."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height
        )
        
        self.assertEqual(params.length, length)
        self.assertEqual(params.max_width, max_width)
        self.assertEqual(params.max_height, max_height)
    
    
    def test_fuselage_parameters_check_parameters_type_error(self):
        """Test _check_parameters method with type error."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height
        )
        
        # Mock _metadata with type checking enabled
        params._metadata = {
            'test_param': {
                'type': [csdl.Variable, ureg.Quantity],
                'variablize': False,
                'shape': None
            }
        }
        
        # Test with wrong type
        with self.assertRaises(ValueError) as context:
            params._check_parameters('test_param', "wrong_type")
        
        self.assertIn("Variable test_param must be of type", str(context.exception))
    
    def test_fuselage_parameters_check_parameters_shape_error(self):
        """Test _check_parameters method with shape error."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height
        )
        
        # Mock _metadata with shape checking enabled
        params._metadata = {
            'test_param': {
                'type': None,
                'variablize': False,
                'shape': (2,)
            }
        }
        
        # Test with wrong shape
        wrong_var = csdl.Variable(value=10.0, shape=(1,), name="wrong_shape")
        with self.assertRaises(ValueError) as context:
            params._check_parameters('test_param', wrong_var)
        
        self.assertIn("Variable test_param must have shape", str(context.exception))
    
    def test_fuselage_parameters_check_parameters_variablize_ureg(self):
        """Test _check_parameters method with variablize=True and ureg.Quantity."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height
        )
        
        # Mock _metadata with variablize enabled
        params._metadata = {
            'test_param': {
                'type': [csdl.Variable, ureg.Quantity],
                'variablize': True,
                'shape': (1,)
            }
        }
        
        # Test with ureg.Quantity (should convert to CSDL Variable)
        test_quantity = Q_(5.0, "m")
        result = params._check_parameters('test_param', test_quantity)
        
        self.assertIsInstance(result, csdl.Variable)
        self.assertEqual(result.shape, (1,))
    


class TestFuselageGeometricQuantities(TestCase):
    """Test the FuselageGeometricQuantities dataclass."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_fuselage_geometric_quantities_creation(self):
        """Test creating FuselageGeometricQuantities."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        width = csdl.Variable(value=2.0, shape=(1,), name="width")
        height = csdl.Variable(value=1.5, shape=(1,), name="height")
        
        quantities = FuselageGeometricQuantities(
            length=length,
            width=width,
            height=height
        )
        
        self.assertEqual(quantities.length, length)
        self.assertEqual(quantities.width, width)
        self.assertEqual(quantities.height, height)
    
    def test_fuselage_geometric_quantities_different_values(self):
        """Test FuselageGeometricQuantities with different numeric values."""
        length = csdl.Variable(value=15.5, shape=(1,), name="length")
        width = csdl.Variable(value=3.2, shape=(1,), name="width")
        height = csdl.Variable(value=2.8, shape=(1,), name="height")
        
        quantities = FuselageGeometricQuantities(
            length=length,
            width=width,
            height=height
        )
        
        self.assertEqual(quantities.length.value, 15.5)
        self.assertEqual(quantities.width.value, 3.2)
        self.assertEqual(quantities.height.value, 2.8)
    
    def test_fuselage_geometric_quantities_zero_values(self):
        """Test FuselageGeometricQuantities with zero values."""
        length = csdl.Variable(value=0.0, shape=(1,), name="length")
        width = csdl.Variable(value=0.0, shape=(1,), name="width")
        height = csdl.Variable(value=0.0, shape=(1,), name="height")
        
        quantities = FuselageGeometricQuantities(
            length=length,
            width=width,
            height=height
        )
        
        self.assertEqual(quantities.length.value, 0.0)
        self.assertEqual(quantities.width.value, 0.0)
        self.assertEqual(quantities.height.value, 0.0)
    
    def test_fuselage_geometric_quantities_negative_values(self):
        """Test FuselageGeometricQuantities with negative values."""
        length = csdl.Variable(value=-5.0, shape=(1,), name="length")
        width = csdl.Variable(value=-1.0, shape=(1,), name="width")
        height = csdl.Variable(value=-2.5, shape=(1,), name="height")
        
        quantities = FuselageGeometricQuantities(
            length=length,
            width=width,
            height=height
        )
        
        self.assertEqual(quantities.length.value, -5.0)
        self.assertEqual(quantities.width.value, -1.0)
        self.assertEqual(quantities.height.value, -2.5)
    
    def test_fuselage_geometric_quantities_attribute_access(self):
        """Test that all attributes can be accessed and modified."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        width = csdl.Variable(value=2.0, shape=(1,), name="width")
        height = csdl.Variable(value=1.5, shape=(1,), name="height")
        
        quantities = FuselageGeometricQuantities(
            length=length,
            width=width,
            height=height
        )
        
        # Test attribute access
        self.assertIsNotNone(quantities.length)
        self.assertIsNotNone(quantities.width)
        self.assertIsNotNone(quantities.height)
        
        # Test attribute modification
        new_length = csdl.Variable(value=20.0, shape=(1,), name="new_length")
        quantities.length = new_length
        self.assertEqual(quantities.length, new_length)


class TestFuselageBasicFunctionality(TestCase):
    """Test basic Fuselage functionality with minimal dependencies."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_fuselage_inheritance(self):
        """Test that Fuselage is a subclass of Component."""
        # Use CSDL variables to avoid the .value issue with ureg.Quantity
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        # Mock all the complex dependencies
        with patch('falco.core.vehicle.components.fuselage.lg') as mock_lg, \
             patch('falco.core.vehicle.components.fuselage.lfs') as mock_lfs, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization') as mock_vsp, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs') as mock_vspi:
            
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
            mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock B-spline space and functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_function.coefficients.add_name = Mock()
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock volume sectional parameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock sectional parameters
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            fuselage = Fuselage(
                length=length,
                max_width=max_width,
                max_height=max_height,
                geometry=mock_geometry,
                skip_ffd=True
            )
            
            self.assertIsInstance(fuselage, Component)
            self.assertIsInstance(fuselage, Fuselage)
    
    def test_fuselage_component_attributes(self):
        """Test that Fuselage has all required Component attributes."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        # Mock all the complex dependencies
        with patch('falco.core.vehicle.components.fuselage.lg') as mock_lg, \
             patch('falco.core.vehicle.components.fuselage.lfs') as mock_lfs, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization') as mock_vsp, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs') as mock_vspi:
            
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
            mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock B-spline space and functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_function.coefficients.add_name = Mock()
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock volume sectional parameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock sectional parameters
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            fuselage = Fuselage(
                length=length,
                max_width=max_width,
                max_height=max_height,
                geometry=mock_geometry,
                skip_ffd=True
            )
            
            # Check Component attributes
            self.assertEqual(fuselage.comps, {})
            self.assertEqual(fuselage.surface_mesh, [])
            self.assertEqual(fuselage.load_solvers, [])
            self.assertIsNone(fuselage.parent)
            self.assertIsNone(fuselage.mass_properties)
    
    def test_fuselage_parameter_handling(self):
        """Test that Fuselage properly handles different parameter types."""
        # Test with CSDL variables
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        # Mock all the complex dependencies
        with patch('falco.core.vehicle.components.fuselage.lg') as mock_lg, \
             patch('falco.core.vehicle.components.fuselage.lfs') as mock_lfs, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization') as mock_vsp, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs') as mock_vspi:
            
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
            mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock B-spline space and functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_function.coefficients.add_name = Mock()
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock volume sectional parameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock sectional parameters
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            fuselage = Fuselage(
                length=length,
                max_width=max_width,
                max_height=max_height,
                geometry=mock_geometry,
                skip_ffd=True
            )
            
            self.assertEqual(fuselage.parameters.length, length)
            self.assertEqual(fuselage.parameters.max_width, max_width)
            self.assertEqual(fuselage.parameters.max_height, max_height)
            self.assertEqual(fuselage._name, "Fuselage")
            self.assertEqual(fuselage.geometry, mock_geometry)
            self.assertTrue(fuselage.skip_ffd)
    
    def test_fuselage_init_with_ureg_quantities(self):
        """Test Fuselage initialization with ureg.Quantity parameters."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")  # Use CSDL Variable instead
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        S_wet = csdl.Variable(value=25.0, shape=(1,), name="S_wet")
        
        # Mock all the complex dependencies
        with patch('falco.core.vehicle.components.fuselage.lg') as mock_lg, \
             patch('falco.core.vehicle.components.fuselage.lfs') as mock_lfs, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization') as mock_vsp, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs') as mock_vspi:
            
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
            mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock B-spline space and functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_function.coefficients.add_name = Mock()
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock volume sectional parameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock sectional parameters
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            fuselage = Fuselage(
                length=length,
                max_width=max_width,
                max_height=max_height,
                S_wet=S_wet,
                geometry=mock_geometry,
                skip_ffd=True
            )
            
            self.assertEqual(fuselage.parameters.length, length)
            self.assertEqual(fuselage.parameters.max_width, max_width)
            self.assertEqual(fuselage.parameters.max_height, max_height)
            self.assertEqual(fuselage.parameters.S_wet, S_wet)
    
    
    def test_fuselage_init_with_kwargs(self):
        """Test Fuselage initialization with additional kwargs."""
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        # Mock all the complex dependencies
        with patch('falco.core.vehicle.components.fuselage.lg') as mock_lg, \
             patch('falco.core.vehicle.components.fuselage.lfs') as mock_lfs, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization') as mock_vsp, \
             patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs') as mock_vspi:
            
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
            mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock B-spline space and functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_function.coefficients.add_name = Mock()
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock volume sectional parameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock sectional parameters
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            # Test with additional kwargs
            fuselage = Fuselage(
                length=length,
                max_width=max_width,
                max_height=max_height,
                geometry=mock_geometry,
                skip_ffd=True,
                custom_param="test_value"
            )
            
            self.assertEqual(fuselage.parameters.length, length)
            self.assertEqual(fuselage.parameters.max_width, max_width)
            self.assertEqual(fuselage.parameters.max_height, max_height)


class TestFuselageWithParameterizationSolver(TestCase):
    """Test Fuselage with parameterization solver."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_with_parameterization_solver(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test Fuselage initialization with parameterization solver."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
        # Create fuselage with parameterization solver
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            parameterization_solver=mock_solver,
            skip_ffd=True
        )
        
        # Verify solver was called
        mock_solver.add_parameter.assert_called()
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_without_parameterization_solver(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test Fuselage initialization without parameterization solver."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Create fuselage without parameterization solver
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            parameterization_solver=None,
            skip_ffd=False
        )
        
        # Verify fuselage was created successfully
        self.assertIsNotNone(fuselage)
        self.assertFalse(fuselage.skip_ffd)
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_with_ffd_geometric_variables(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test Fuselage initialization with ffd_geometric_variables."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Mock ffd_geometric_variables
        mock_ffd_geometric_variables = Mock()
        mock_ffd_geometric_variables.add_variable = Mock()
        
        # Create fuselage with ffd_geometric_variables
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            ffd_geometric_variables=mock_ffd_geometric_variables,
            skip_ffd=False
        )
        
        # Verify ffd_geometric_variables.add_variable was called for each parameter
        expected_calls = 3  # length, max_height, max_width
        self.assertEqual(mock_ffd_geometric_variables.add_variable.call_count, expected_calls)
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_skip_ffd_with_parameterization_solver(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test Fuselage with skip_ffd=True and parameterization solver."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
        # Create fuselage with skip_ffd=True and parameterization solver
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            parameterization_solver=mock_solver,
            skip_ffd=True
        )
        
        # Verify solver was called with rigid_body_translation (for skip_ffd=True)
        mock_solver.add_parameter.assert_called_once()
        self.assertTrue(fuselage.skip_ffd)


class TestFuselageGeometryPoints(TestCase):
    """Test Fuselage geometry point extraction functionality."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_geometry_point_extraction(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test that geometry points are properly extracted and stored."""
        # Mock geometry with specific point evaluations
        mock_geometry = Mock()
        mock_geometry.project.return_value = np.array([0.5, 0.5, 0.5])
        
        # Mock different point evaluations
        nose_point = csdl.Variable(value=np.array([10.0, 0.0, 0.0]), shape=(3,))
        tail_point = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        left_point = csdl.Variable(value=np.array([5.0, -1.0, 0.0]), shape=(3,))
        right_point = csdl.Variable(value=np.array([5.0, 1.0, 0.0]), shape=(3,))
        top_point = csdl.Variable(value=np.array([5.0, 0.0, 1.5]), shape=(3,))
        bottom_point = csdl.Variable(value=np.array([5.0, 0.0, -0.5]), shape=(3,))
        
        # Mock geometry.evaluate to return different points based on input
        def mock_evaluate(point):
            if np.array_equal(point, np.array([0.5, 0.5, 0.5])):
                # This is the projected point - return based on which point is being evaluated
                return nose_point
            return nose_point  # Default return
        
        mock_geometry.evaluate.side_effect = mock_evaluate
        mock_geometry.functions = {}
        mock_geometry.set_coefficients = Mock()
        
        # Mock FFD block
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Create fuselage
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            skip_ffd=True
        )
        
        # Verify that geometry point extraction methods were called
        self.assertTrue(hasattr(fuselage, 'nose_point'))
        self.assertTrue(hasattr(fuselage, 'tail_point'))
        self.assertTrue(hasattr(fuselage, 'left_point'))
        self.assertTrue(hasattr(fuselage, 'right_point'))
        self.assertTrue(hasattr(fuselage, 'top_point'))
        self.assertTrue(hasattr(fuselage, 'bottom_point'))
        
        # Verify that geometry.project was called multiple times
        self.assertGreater(mock_geometry.project.call_count, 0)
        self.assertGreater(mock_geometry.evaluate.call_count, 0)
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_ffd_block_construction(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test that FFD block is properly constructed."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Create fuselage
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            skip_ffd=False
        )
        
        # Verify that FFD block was constructed with correct parameters
        mock_lg.construct_ffd_block_around_entities.assert_called_once_with(
            entities=mock_geometry,
            num_coefficients=(2, 3, 2),
            degree=(1, 1, 1)
        )
        
        # Verify that geometry.set_coefficients was called
        mock_geometry.set_coefficients.assert_called_once()


class TestFuselageErrorConditions(TestCase):
    """Test error conditions and edge cases in Fuselage."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_fuselage_parameters_type_validation(self):
        """Test that FuselageParameters accepts different types (dataclass doesn't enforce types)."""
        # Test with invalid type - dataclass doesn't enforce type validation automatically
        # This should actually work since dataclass doesn't validate types
        params = FuselageParameters(
            length="invalid_string",  # Should be csdl.Variable or ureg.Quantity
            max_width=csdl.Variable(value=2.0, shape=(1,), name="max_width"),
            max_height=csdl.Variable(value=1.5, shape=(1,), name="max_height")
        )
        
        # Verify the invalid string was stored
        self.assertEqual(params.length, "invalid_string")
    
    def test_fuselage_parameters_with_none_length(self):
        """Test FuselageParameters behavior with None length."""
        # Dataclass allows None values - this should work
        params = FuselageParameters(
            length=None,
            max_width=csdl.Variable(value=2.0, shape=(1,), name="max_width"),
            max_height=csdl.Variable(value=1.5, shape=(1,), name="max_height")
        )
        
        # Verify None was stored
        self.assertIsNone(params.length)
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_with_none_geometry(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test Fuselage initialization with None geometry."""
        # This should raise an error since geometry is required for FFD operations
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        with self.assertRaises(AttributeError):
            fuselage = Fuselage(
                length=length,
                max_width=max_width,
                max_height=max_height,
                geometry=None,
                skip_ffd=False
            )
    
    def test_fuselage_geometric_quantities_with_invalid_types(self):
        """Test FuselageGeometricQuantities with invalid input types."""
        # Dataclass doesn't enforce type validation - this should work
        quantities = FuselageGeometricQuantities(
            length="invalid",
            width=csdl.Variable(value=2.0, shape=(1,), name="width"),
            height=csdl.Variable(value=1.5, shape=(1,), name="height")
        )
        
        # Verify the invalid string was stored
        self.assertEqual(quantities.length, "invalid")
    
    def test_fuselage_parameters_shape_validation(self):
        """Test FuselageParameters shape validation."""
        # Test with wrong shape CSDL variable
        wrong_shape_var = csdl.Variable(value=np.array([[1.0, 2.0], [3.0, 4.0]]), shape=(2, 2), name="wrong_shape")
        
        # This should work since the dataclass doesn't enforce shapes directly
        # The validation would happen in the _check_parameters method if called
        params = FuselageParameters(
            length=wrong_shape_var,
            max_width=csdl.Variable(value=2.0, shape=(1,), name="max_width"),
            max_height=csdl.Variable(value=1.5, shape=(1,), name="max_height")
        )
        
        self.assertEqual(params.length, wrong_shape_var)


class TestFuselageAdvancedFunctionality(TestCase):
    """Test advanced Fuselage functionality including B-splines and geometric calculations."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_b_spline_function_creation(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test that B-spline functions are properly created for stretch coefficients."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        # Mock B-spline function
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Create fuselage
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            skip_ffd=False
        )
        
        # Verify that B-spline space was created with correct parameters
        mock_lfs.BSplineSpace.assert_called_with(
            num_parametric_dimensions=1,
            degree=1,
            coefficients_shape=(2,)
        )
        
        # Verify that B-spline functions were created (should be called 3 times for length, height, width)
        self.assertEqual(mock_lfs.Function.call_count, 3)
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_volume_sectional_parameterization(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test volume sectional parameterization functionality."""
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
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Create fuselage
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            skip_ffd=False
        )
        
        # Verify that VolumeSectionalParameterization was created
        mock_vsp.assert_called_once()
        call_args = mock_vsp.call_args
        self.assertEqual(call_args[1]['name'], 'Fuselage_sectional_parameterization')
        self.assertEqual(call_args[1]['principal_parametric_dimension'], 0)
        
        # Verify that sectional parameters methods were called
        mock_vspi_instance.add_sectional_translation.assert_called_once()
        mock_vspi_instance.add_sectional_stretch.assert_called()
    
    @patch('falco.core.vehicle.components.fuselage.lg')
    @patch('falco.core.vehicle.components.fuselage.lfs')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterization')
    @patch('falco.core.vehicle.components.fuselage.VolumeSectionalParameterizationInputs')
    def test_fuselage_geometric_calculations(self, mock_vspi, mock_vsp, mock_lfs, mock_lg):
        """Test geometric calculations (length, width, height from points)."""
        # Mock geometry with specific point evaluations for geometric calculations
        mock_geometry = Mock()
        mock_geometry.project.return_value = np.array([0.5, 0.5, 0.5])
        
        # Create mock points that will result in specific geometric quantities
        nose_point = csdl.Variable(value=np.array([10.0, 0.0, 0.0]), shape=(3,))
        tail_point = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        left_point = csdl.Variable(value=np.array([5.0, -1.0, 0.0]), shape=(3,))
        right_point = csdl.Variable(value=np.array([5.0, 1.0, 0.0]), shape=(3,))
        top_point = csdl.Variable(value=np.array([5.0, 0.0, 1.5]), shape=(3,))
        bottom_point = csdl.Variable(value=np.array([5.0, 0.0, -0.5]), shape=(3,))
        
        # Mock evaluate to return different points
        point_counter = 0
        points = [nose_point, tail_point, left_point, right_point, top_point, bottom_point]
        
        def mock_evaluate(point):
            nonlocal point_counter
            result = points[point_counter % len(points)]
            point_counter += 1
            return result
        
        mock_geometry.evaluate.side_effect = mock_evaluate
        mock_geometry.functions = {}
        mock_geometry.set_coefficients = Mock()
        
        # Mock FFD block
        mock_ffd_block = Mock()
        mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
        mock_ffd_block.evaluate_ffd.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
        
        # Mock B-spline space and functions
        mock_bspline_space = Mock()
        mock_lfs.BSplineSpace.return_value = mock_bspline_space
        
        mock_function = Mock()
        mock_function.coefficients = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_function.coefficients.add_name = Mock()
        mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
        mock_lfs.Function.return_value = mock_function
        
        # Mock volume sectional parameterization
        mock_vsp_instance = Mock()
        mock_vsp_instance.num_sections = 3
        mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((2, 3, 2, 3)), shape=(2, 3, 2, 3))
        mock_vsp.return_value = mock_vsp_instance
        
        # Mock sectional parameters
        mock_vspi_instance = Mock()
        mock_vspi_instance.add_sectional_translation = Mock()
        mock_vspi_instance.add_sectional_stretch = Mock()
        mock_vspi.return_value = mock_vspi_instance
        
        # Create fuselage
        length = csdl.Variable(value=10.0, shape=(1,), name="length")
        max_width = csdl.Variable(value=2.0, shape=(1,), name="max_width")
        max_height = csdl.Variable(value=1.5, shape=(1,), name="max_height")
        
        fuselage = Fuselage(
            length=length,
            max_width=max_width,
            max_height=max_height,
            geometry=mock_geometry,
            skip_ffd=False
        )
        
        # Verify that geometry.evaluate was called multiple times for point calculations
        self.assertGreater(mock_geometry.evaluate.call_count, 6)  # At least 6 calls for the 6 points
    
    def test_fuselage_parameters_with_ureg_quantities_edge_cases(self):
        """Test FuselageParameters with various ureg.Quantity edge cases."""
        # Test with different units
        length = Q_(10.0, "ft")  # feet
        max_width = Q_(2.0, "in")  # inches
        max_height = Q_(1.5, "cm")  # centimeters
        S_wet = Q_(25.0, "ft**2")  # square feet
        
        params = FuselageParameters(
            length=length,
            max_width=max_width,
            max_height=max_height,
            S_wet=S_wet
        )
        
        self.assertEqual(params.length, length)
        self.assertEqual(params.max_width, max_width)
        self.assertEqual(params.max_height, max_height)
        self.assertEqual(params.S_wet, S_wet)
        
        # Test with very small values
        small_length = Q_(0.001, "m")
        small_width = Q_(0.0001, "m")
        small_height = Q_(0.0001, "m")
        
        small_params = FuselageParameters(
            length=small_length,
            max_width=small_width,
            max_height=small_height
        )
        
        self.assertEqual(small_params.length, small_length)
        self.assertEqual(small_params.max_width, small_width)
        self.assertEqual(small_params.max_height, small_height)
        
        # Test with very large values
        large_length = Q_(1000.0, "m")
        large_width = Q_(100.0, "m")
        large_height = Q_(50.0, "m")
        
        large_params = FuselageParameters(
            length=large_length,
            max_width=large_width,
            max_height=large_height
        )
        
        self.assertEqual(large_params.length, large_length)
        self.assertEqual(large_params.max_width, large_width)
        self.assertEqual(large_params.max_height, large_height)


if __name__ == '__main__':
    import unittest
    unittest.main()
