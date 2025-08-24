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


if __name__ == '__main__':
    import unittest
    unittest.main()
