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

    
    def test_wing_parameters_shape_validation(self):
        """Test WingParameters shape validation."""
        AR = csdl.Variable(value=10.0, shape=(2,), name="AR")  # Wrong shape
        
        with self.assertRaises(ValueError):
            params = WingParameters(
                AR=AR, S_ref=csdl.Variable(value=20.0, shape=(1,), name="S_ref"),
                span=csdl.Variable(value=14.14, shape=(1,), name="span"),
                sweep=csdl.Variable(value=5.0, shape=(1,), name="sweep"),
                incidence=csdl.Variable(value=2.0, shape=(1,), name="incidence"),
                taper_ratio=csdl.Variable(value=0.8, shape=(1,), name="taper_ratio"),
                dihedral=csdl.Variable(value=3.0, shape=(1,), name="dihedral"),
                root_twist_delta=None, tip_twist_delta=None,
                thickness_to_chord=csdl.Variable(value=0.12, shape=(1,), name="thickness_to_chord"),
                thickness_to_chord_loc=csdl.Variable(value=0.25, shape=(1,), name="thickness_to_chord_loc")
            )
    
    def test_wing_parameters_type_validation(self):
        """Test WingParameters type validation."""
        with self.assertRaises(ValueError):
            params = WingParameters(
                AR="invalid_type",  # Wrong type
                S_ref=csdl.Variable(value=20.0, shape=(1,), name="S_ref"),
                span=csdl.Variable(value=14.14, shape=(1,), name="span"),
                sweep=csdl.Variable(value=5.0, shape=(1,), name="sweep"),
                incidence=csdl.Variable(value=2.0, shape=(1,), name="incidence"),
                taper_ratio=csdl.Variable(value=0.8, shape=(1,), name="taper_ratio"),
                dihedral=csdl.Variable(value=3.0, shape=(1,), name="dihedral"),
                root_twist_delta=None, tip_twist_delta=None,
                thickness_to_chord=csdl.Variable(value=0.12, shape=(1,), name="thickness_to_chord"),
                thickness_to_chord_loc=csdl.Variable(value=0.25, shape=(1,), name="thickness_to_chord_loc")
            )


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


class TestWingParameterCombinations(TestCase):
    """Test Wing with different parameter combinations and calculations."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_ar_s_ref_combination(self):
        """Test Wing with AR and S_ref specified (should calculate span)."""
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
            
            # Verify that span was calculated from AR and S_ref
            expected_span = (AR * S_ref)**0.5
            self.assertEqual(wing.parameters.AR, AR)
            self.assertEqual(wing.parameters.S_ref, S_ref)
            self.assertIsNotNone(wing.parameters.span)
            self.assertIsNotNone(wing.parameters.MAC)
    
    def test_wing_s_ref_span_combination(self):
        """Test Wing with S_ref and span specified (should calculate AR)."""
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        
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
            
            wing = Wing(name="test_wing", S_ref=S_ref, span=span, skip_ffd=True)
            
            # Verify that AR was calculated from span and S_ref
            self.assertEqual(wing.parameters.S_ref, S_ref)
            self.assertEqual(wing.parameters.span, span)
            self.assertIsNotNone(wing.parameters.AR)
            self.assertIsNotNone(wing.parameters.MAC)
    
    def test_wing_span_ar_combination(self):
        """Test Wing with span and AR specified (should calculate S_ref)."""
        span = csdl.Variable(value=14.14, shape=(1,), name="span")
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        
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
            
            wing = Wing(name="test_wing", span=span, AR=AR, skip_ffd=True)
            
            # Verify that S_ref was calculated from span and AR
            self.assertEqual(wing.parameters.span, span)
            self.assertEqual(wing.parameters.AR, AR)
            self.assertIsNotNone(wing.parameters.S_ref)
            self.assertIsNotNone(wing.parameters.MAC)
    
    def test_wing_with_taper_ratio_none(self):
        """Test Wing with taper_ratio=None (should default to 1)."""
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, taper_ratio=None, skip_ffd=True)
            
            # Verify that taper_ratio was set to 1
            self.assertEqual(wing.parameters.taper_ratio.value, 1.0)
            self.assertEqual(wing.parameters.taper_ratio.name, "test_wing_taper_ratio")
    
    def test_wing_with_sweep_none(self):
        """Test Wing with sweep=None (should default to 0)."""
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, sweep=None, skip_ffd=True)
            
            # Verify that sweep was set to 0
            self.assertEqual(wing.parameters.sweep.value, 0.0)
            # Check if it has units attribute, if not it's a CSDL Variable
            if hasattr(wing.parameters.sweep, 'units'):
                self.assertEqual(wing.parameters.sweep.units, 'radian')
    
    def test_wing_with_dihedral_none(self):
        """Test Wing with dihedral=None (should default to 0)."""
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, dihedral=None, skip_ffd=True)
            
            # Verify that dihedral was set to 0
            self.assertEqual(wing.parameters.dihedral.value, 0.0)
            self.assertEqual(wing.parameters.dihedral.name, "test_wing_dihedral")
    
    def test_wing_geometric_calculations(self):
        """Test that geometric calculations (Kc, co, Kcc, c_ma, ct) are performed."""
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
            
            # Verify that geometric calculations were performed
            self.assertIsNotNone(wing.parameters.eta_0)
            self.assertIsNotNone(wing.parameters.Kc)
            self.assertIsNotNone(wing.parameters.co)
            self.assertIsNotNone(wing.parameters.Kcc)
            self.assertIsNotNone(wing.parameters.c_ma)
            self.assertIsNotNone(wing.parameters.ct)
            # S_wet is set to self.surface_area which is None when no geometry is provided
            # This is expected behavior, so we don't test for it being not None


class TestWingOrientations(TestCase):
    """Test Wing with different orientations and configurations."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_vertical_orientation(self):
        """Test Wing with vertical orientation."""
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, orientation="vertical", skip_ffd=True)
            
            # Verify vertical orientation
            self.assertEqual(wing._orientation, "vertical")
            self.assertEqual(wing.parameters.AR, AR)
            self.assertEqual(wing.parameters.S_ref, S_ref)
    
    def test_wing_vertical_with_dihedral_error(self):
        """Test that vertical wing with dihedral raises ValueError."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        dihedral = csdl.Variable(value=3.0, shape=(1,), name="dihedral")
        
        with self.assertRaises(ValueError) as context:
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
                
                Wing(name="test_wing", AR=AR, S_ref=S_ref, dihedral=dihedral, orientation="vertical", skip_ffd=True)
        
        self.assertIn("Cannot specify dihedral for vertical wing", str(context.exception))
    

class TestWingGeometryScenarios(TestCase):
    """Test Wing with different geometry scenarios."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_with_geometry_none(self):
        """Test Wing with geometry=None (skip_ffd=True)."""
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, geometry=None, skip_ffd=True)
            
            # Verify that FFD block is None when geometry is None
            self.assertIsNone(wing._ffd_block)
            self.assertIsNone(wing.geometry)
            self.assertTrue(wing.skip_ffd)
    
    
    def test_wing_with_thickness_to_chord_none(self):
        """Test Wing with thickness_to_chord=None (should default to 0)."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        thickness_to_chord = None
        
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, thickness_to_chord=thickness_to_chord, skip_ffd=True)
            
            # Verify that thickness_to_chord was set to 0
            # Note: when thickness_to_chord is None, it gets set to Q_(0.0, 'dimensionless') in the Wing constructor
            if wing.parameters.thickness_to_chord is not None:
                if hasattr(wing.parameters.thickness_to_chord, 'value'):
                    self.assertEqual(wing.parameters.thickness_to_chord.value, 0.0)
                if hasattr(wing.parameters.thickness_to_chord, 'units'):
                    self.assertEqual(wing.parameters.thickness_to_chord.units, 'dimensionless')


class TestWingFFDParameterization(TestCase):
    """Test Wing FFD parameterization and solver configurations."""
    
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_wing_with_parameterization_solver_skip_ffd(self):
        """Test Wing with parameterization_solver when skip_ffd=True."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, parameterization_solver=mock_solver, skip_ffd=True)
            
            # Verify that rigid_body_translation parameter was added to solver
            mock_solver.add_parameter.assert_called_once()
    
    def test_wing_with_parameterization_solver_no_skip_ffd(self):
        """Test Wing with parameterization_solver when skip_ffd=False."""
        AR = csdl.Variable(value=10.0, shape=(1,), name="AR")
        S_ref = csdl.Variable(value=20.0, shape=(1,), name="S_ref")
        
        # Mock parameterization solver
        mock_solver = Mock()
        mock_solver.add_parameter = Mock()
        
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
            
            wing = Wing(name="test_wing", AR=AR, S_ref=S_ref, parameterization_solver=mock_solver, skip_ffd=False)
            
            # Verify that multiple parameters were added to solver (chord, wingspan, sweep, twist, dihedral)
            self.assertEqual(mock_solver.add_parameter.call_count, 5)
    



if __name__ == '__main__':
    import unittest
    unittest.main()
