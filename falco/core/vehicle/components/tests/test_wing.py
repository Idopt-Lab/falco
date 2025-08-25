from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from falco import ureg, Q_
import csdl_alpha as csdl

from falco.core.vehicle.components.wing import Wing, WingParameters, WingGeometricQuantities
from falco.core.vehicle.components.component import Component


class TestWingParameters(TestCase):
    """Test the WingParameters dataclass."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_parameters_creation(self):
        """Test creating WingParameters with different input types."""
        # Test with basic parameters
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        sweep = csdl.Variable(value=5.0, shape=(1,), name="sweep")
        incidence = csdl.Variable(value=2.0, shape=(1,), name="incidence")
        taper_ratio = csdl.Variable(value=0.8, shape=(1,), name="taper_ratio")
        dihedral = csdl.Variable(value=3.0, shape=(1,), name="dihedral")
        root_twist_delta = csdl.Variable(value=1.0, shape=(1,), name="root_twist_delta")
        tip_twist_delta = csdl.Variable(value=-1.0, shape=(1,), name="tip_twist_delta")
        thickness_to_chord = csdl.Variable(value=0.12, shape=(1,), name="thickness_to_chord")
        thickness_to_chord_loc = csdl.Variable(value=0.25, shape=(1,), name="thickness_to_chord_loc")
        
        params = WingParameters(
            AR=AR, S_ref=S_ref, span=span, sweep=sweep, incidence=incidence,
            taper_ratio=taper_ratio, dihedral=dihedral, root_twist_delta=root_twist_delta,
            tip_twist_delta=tip_twist_delta, thickness_to_chord=thickness_to_chord,
            thickness_to_chord_loc=thickness_to_chord_loc
        )
        
        self.assertEqual(params.AR, AR)
        self.assertEqual(params.S_ref, S_ref)
        self.assertEqual(params.span, span)
        self.assertEqual(params.sweep, sweep)
        self.assertEqual(params.incidence, incidence)
        self.assertEqual(params.taper_ratio, taper_ratio)
        self.assertEqual(params.dihedral, dihedral)
        self.assertEqual(params.root_twist_delta, root_twist_delta)
        self.assertEqual(params.tip_twist_delta, tip_twist_delta)
        self.assertEqual(params.thickness_to_chord, thickness_to_chord)
        self.assertEqual(params.thickness_to_chord_loc, thickness_to_chord_loc)
    
    def test_wing_parameters_with_optional_fields(self):
        """Test WingParameters with optional fields."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        sweep = csdl.Variable(value=5.0, shape=(1,), name="sweep")
        incidence = csdl.Variable(value=2.0, shape=(1,), name="incidence")
        taper_ratio = csdl.Variable(value=0.8, shape=(1,), name="taper_ratio")
        dihedral = csdl.Variable(value=3.0, shape=(1,), name="dihedral")
        thickness_to_chord = csdl.Variable(value=0.12, shape=(1,), name="thickness_to_chord")
        thickness_to_chord_loc = csdl.Variable(value=0.25, shape=(1,), name="thickness_to_chord_loc")
        
        # Test with None values for optional fields
        params = WingParameters(
            AR=AR, S_ref=S_ref, span=span, sweep=sweep, incidence=incidence,
            taper_ratio=taper_ratio, dihedral=dihedral, root_twist_delta=None,
            tip_twist_delta=None, thickness_to_chord=thickness_to_chord,
            thickness_to_chord_loc=thickness_to_chord_loc, actuate_angle=None,
            actuate_axis_location=None, MAC=None, S_wet=None, eta_0=None,
            Kc=None, co=None, Kcc=None, c_ma=None, ct=None
        )
        
        self.assertIsNone(params.root_twist_delta)
        self.assertIsNone(params.tip_twist_delta)
        self.assertIsNone(params.actuate_angle)
        self.assertIsNone(params.actuate_axis_location)
        self.assertIsNone(params.MAC)
        self.assertIsNone(params.S_wet)
        self.assertIsNone(params.eta_0)
        self.assertIsNone(params.Kc)
        self.assertIsNone(params.co)
        self.assertIsNone(params.Kcc)
        self.assertIsNone(params.c_ma)
        self.assertIsNone(params.ct)
    
    def test_wing_parameters_define_checks(self):
        """Test the define_checks method."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        sweep = csdl.Variable(value=5.0, shape=(1,), name="sweep")
        incidence = csdl.Variable(value=2.0, shape=(1,), name="incidence")
        taper_ratio = csdl.Variable(value=0.8, shape=(1,), name="taper_ratio")
        dihedral = csdl.Variable(value=3.0, shape=(1,), name="dihedral")
        thickness_to_chord = csdl.Variable(value=0.12, shape=(1,), name="thickness_to_chord")
        thickness_to_chord_loc = csdl.Variable(value=0.25, shape=(1,), name="thickness_to_chord_loc")
        
        params = WingParameters(
            AR=AR, S_ref=S_ref, span=span, sweep=sweep, incidence=incidence,
            taper_ratio=taper_ratio, dihedral=dihedral, root_twist_delta=None,
            tip_twist_delta=None, thickness_to_chord=thickness_to_chord,
            thickness_to_chord_loc=thickness_to_chord_loc
        )
        
        # Test that define_checks method exists and can be called
        self.assertTrue(hasattr(params, 'define_checks'))
        self.assertTrue(callable(params.define_checks))
    
    def test_wing_parameters_check_parameters(self):
        """Test the _check_parameters method."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        sweep = csdl.Variable(value=5.0, shape=(1,), name="sweep")
        incidence = csdl.Variable(value=2.0, shape=(1,), name="incidence")
        taper_ratio = csdl.Variable(value=0.8, shape=(1,), name="taper_ratio")
        dihedral = csdl.Variable(value=3.0, shape=(1,), name="dihedral")
        thickness_to_chord = csdl.Variable(value=0.12, shape=(1,), name="thickness_to_chord")
        thickness_to_chord_loc = csdl.Variable(value=0.25, shape=(1,), name="thickness_to_chord_loc")
        
        params = WingParameters(
            AR=AR, S_ref=S_ref, span=span, sweep=sweep, incidence=incidence,
            taper_ratio=taper_ratio, dihedral=dihedral, root_twist_delta=None,
            tip_twist_delta=None, thickness_to_chord=thickness_to_chord,
            thickness_to_chord_loc=thickness_to_chord_loc
        )
        
        # Test that _check_parameters method exists and can be called
        self.assertTrue(hasattr(params, '_check_parameters'))
        self.assertTrue(callable(params._check_parameters))
        
        # Test with valid parameters
        result = params._check_parameters('AR', AR)
        self.assertEqual(result, AR)
        
        # Test with None value
        result = params._check_parameters('root_twist_delta', None)
        self.assertIsNone(result)


class TestWingGeometricQuantities(TestCase):
    """Test the WingGeometricQuantities dataclass."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_geometric_quantities_creation(self):
        """Test creating WingGeometricQuantities."""
        span = csdl.Variable(value=10.0, shape=(1,), name="span")
        center_chord = csdl.Variable(value=2.0, shape=(1,), name="center_chord")
        left_tip_chord = csdl.Variable(value=1.5, shape=(1,), name="left_tip_chord")
        right_tip_chord = csdl.Variable(value=1.5, shape=(1,), name="right_tip_chord")
        sweep_angle_left = csdl.Variable(value=5.0, shape=(1,), name="sweep_angle_left")
        sweep_angle_right = csdl.Variable(value=5.0, shape=(1,), name="sweep_angle_right")
        dihedral_angle_left = csdl.Variable(value=3.0, shape=(1,), name="dihedral_angle_left")
        dihedral_angle_right = csdl.Variable(value=3.0, shape=(1,), name="dihedral_angle_right")
        
        quantities = WingGeometricQuantities(
            span=span, center_chord=center_chord, left_tip_chord=left_tip_chord,
            right_tip_chord=right_tip_chord, sweep_angle_left=sweep_angle_left,
            sweep_angle_right=sweep_angle_right, dihedral_angle_left=dihedral_angle_left,
            dihedral_angle_right=dihedral_angle_right
        )
        
        self.assertEqual(quantities.span, span)
        self.assertEqual(quantities.center_chord, center_chord)
        self.assertEqual(quantities.left_tip_chord, left_tip_chord)
        self.assertEqual(quantities.right_tip_chord, right_tip_chord)
        self.assertEqual(quantities.sweep_angle_left, sweep_angle_left)
        self.assertEqual(quantities.sweep_angle_right, sweep_angle_right)
        self.assertEqual(quantities.dihedral_angle_left, dihedral_angle_left)
        self.assertEqual(quantities.dihedral_angle_right, dihedral_angle_right)


class TestWingBasicFunctionality(TestCase):
    """Test basic Wing functionality with minimal dependencies."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_inheritance(self):
        """Test that Wing is a subclass of Component."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        
        # Mock all the complex FFD operations
        with patch('falco.core.vehicle.components.wing.lg') as mock_lg, \
             patch('falco.core.vehicle.components.wing.lfs') as mock_lfs, \
             patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterization') as mock_vsp, \
             patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterizationInputs') as mock_vspi, \
             patch('falco.core.vehicle.components.wing.csdl.expand') as mock_expand, \
             patch('falco.core.vehicle.components.wing.csdl.norm') as mock_norm, \
             patch('falco.core.vehicle.components.wing.csdl.arcsin') as mock_arcsin, \
             patch('falco.core.vehicle.components.wing.csdl.linear_combination') as mock_linear_combination:
            
            # Mock FFD block
            mock_ffd_block = Mock()
            mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            
            # Mock lg.construct_ffd_block_around_entities
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock lfs functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock csdl functions
            mock_expand.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
            mock_arcsin.return_value = csdl.Variable(value=0.1, shape=(1,))
            mock_linear_combination.return_value = Mock()
            
            # Mock VolumeSectionalParameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock VolumeSectionalParameterizationInputs
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_rotation = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, skip_ffd=True)
            
            self.assertIsInstance(wing, Component)
            self.assertIsInstance(wing, Wing)
    
    def test_wing_parameter_handling(self):
        """Test that Wing properly handles different parameter types."""
        # Test with CSDL variables
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        
        # Mock all the complex FFD operations
        with patch('falco.core.vehicle.components.wing.lg') as mock_lg, \
             patch('falco.core.vehicle.components.wing.lfs') as mock_lfs, \
             patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterization') as mock_vsp, \
             patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterizationInputs') as mock_vspi, \
             patch('falco.core.vehicle.components.wing.csdl.expand') as mock_expand, \
             patch('falco.core.vehicle.components.wing.csdl.norm') as mock_norm, \
             patch('falco.core.vehicle.components.wing.csdl.arcsin') as mock_arcsin, \
             patch('falco.core.vehicle.components.wing.csdl.linear_combination') as mock_linear_combination:
            
            # Mock FFD block
            mock_ffd_block = Mock()
            mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
            
            # Mock lg.construct_ffd_block_around_entities
            mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
            
            # Mock lfs functions
            mock_bspline_space = Mock()
            mock_lfs.BSplineSpace.return_value = mock_bspline_space
            
            mock_function = Mock()
            mock_function.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
            mock_lfs.Function.return_value = mock_function
            
            # Mock csdl functions
            mock_expand.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
            mock_arcsin.return_value = csdl.Variable(value=0.1, shape=(1,))
            mock_linear_combination.return_value = Mock()
            
            # Mock VolumeSectionalParameterization
            mock_vsp_instance = Mock()
            mock_vsp_instance.num_sections = 3
            mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
            mock_vsp.return_value = mock_vsp_instance
            
            # Mock VolumeSectionalParameterizationInputs
            mock_vspi_instance = Mock()
            mock_vspi_instance.add_sectional_stretch = Mock()
            mock_vspi_instance.add_sectional_translation = Mock()
            mock_vspi_instance.add_sectional_rotation = Mock()
            mock_vspi.return_value = mock_vspi_instance
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, skip_ffd=True)
            
            self.assertEqual(wing.parameters.AR, AR)
            self.assertEqual(wing.parameters.S_ref, S_ref)
            self.assertEqual(wing._name, "test_wing")
            self.assertIsNone(wing.geometry)
            self.assertTrue(wing.skip_ffd)


class TestWingParameterValidation(TestCase):
    """Test Wing parameter validation and error handling."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_over_parameterized(self):
        """Test that Wing raises exception when over-parameterized."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        
        with self.assertRaises(Exception) as context:
            # Mock all the complex FFD operations
            with patch('falco.core.vehicle.components.wing.lg') as mock_lg, \
                 patch('falco.core.vehicle.components.wing.lfs') as mock_lfs, \
                 patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterization') as mock_vsp, \
                 patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterizationInputs') as mock_vspi, \
                 patch('falco.core.vehicle.components.wing.csdl.expand') as mock_expand, \
                 patch('falco.core.vehicle.components.wing.csdl.norm') as mock_norm, \
                 patch('falco.core.vehicle.components.wing.csdl.arcsin') as mock_arcsin, \
                 patch('falco.core.vehicle.components.wing.csdl.linear_combination') as mock_linear_combination:
                
                # Mock FFD block
                mock_ffd_block = Mock()
                mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
                
                # Mock lg.construct_ffd_block_around_entities
                mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
                
                # Mock lfs functions
                mock_bspline_space = Mock()
                mock_lfs.BSplineSpace.return_value = mock_bspline_space
                
                mock_function = Mock()
                mock_function.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
                mock_lfs.Function.return_value = mock_function
                
                # Mock csdl functions
                mock_expand.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
                mock_arcsin.return_value = csdl.Variable(value=0.1, shape=(1,))
                mock_linear_combination.return_value = Mock()
                
                # Mock VolumeSectionalParameterization
                mock_vsp_instance = Mock()
                mock_vsp_instance.num_sections = 3
                mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_vsp.return_value = mock_vsp_instance
                
                # Mock VolumeSectionalParameterizationInputs
                mock_vspi_instance = Mock()
                mock_vspi_instance.add_sectional_stretch = Mock()
                mock_vspi_instance.add_sectional_translation = Mock()
                mock_vspi_instance.add_sectional_rotation = Mock()
                mock_vspi.return_value = mock_vspi_instance
                
                Wing(name="test_wing", AR=AR, S_ref=S_ref, span=span, skip_ffd=True)
        
        self.assertIn("over-parameterized", str(context.exception))
    
    def test_wing_under_parameterized(self):
        """Test that Wing raises exception when under-parameterized."""
        # Test with only AR specified
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        
        with self.assertRaises(Exception) as context:
            # Mock all the complex FFD operations
            with patch('falco.core.vehicle.components.wing.lg') as mock_lg, \
                 patch('falco.core.vehicle.components.wing.lfs') as mock_lfs, \
                 patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterization') as mock_vsp, \
                 patch('lsdo_geo.core.parameterization.volume_sectional_parameterization.VolumeSectionalParameterizationInputs') as mock_vspi, \
                 patch('falco.core.vehicle.components.wing.csdl.expand') as mock_expand, \
                 patch('falco.core.vehicle.components.wing.csdl.norm') as mock_norm, \
                 patch('falco.core.vehicle.components.wing.csdl.arcsin') as mock_arcsin, \
                 patch('falco.core.vehicle.components.wing.csdl.linear_combination') as mock_linear_combination:
                
                # Mock FFD block
                mock_ffd_block = Mock()
                mock_ffd_block.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_ffd_block.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0, 0.0]), shape=(3,))
                
                # Mock lg.construct_ffd_block_around_entities
                mock_lg.construct_ffd_block_around_entities.return_value = mock_ffd_block
                
                # Mock lfs functions
                mock_bspline_space = Mock()
                mock_lfs.BSplineSpace.return_value = mock_bspline_space
                
                mock_function = Mock()
                mock_function.coefficients = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_function.evaluate.return_value = csdl.Variable(value=np.array([0.0, 0.0]), shape=(2,))
                mock_lfs.Function.return_value = mock_function
                
                # Mock csdl functions
                mock_expand.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_norm.return_value = csdl.Variable(value=1.0, shape=(1,))
                mock_arcsin.return_value = csdl.Variable(value=0.1, shape=(1,))
                mock_linear_combination.return_value = Mock()
                
                # Mock VolumeSectionalParameterization
                mock_vsp_instance = Mock()
                mock_vsp_instance.num_sections = 3
                mock_vsp_instance.evaluate.return_value = csdl.Variable(value=np.zeros((3, 11, 3)), shape=(3, 11, 3))
                mock_vsp.return_value = mock_vsp_instance
                
                # Mock VolumeSectionalParameterizationInputs
                mock_vspi_instance = Mock()
                mock_vspi_instance.add_sectional_stretch = Mock()
                mock_vspi_instance.add_sectional_translation = Mock()
                mock_vspi_instance.add_sectional_rotation = Mock()
                mock_vspi.return_value = mock_vspi_instance
                
                Wing(name="test_wing", AR=AR, skip_ffd=True)
        
        self.assertIn("under-parameterized", str(context.exception))


if __name__ == '__main__':
    import unittest
    unittest.main()
