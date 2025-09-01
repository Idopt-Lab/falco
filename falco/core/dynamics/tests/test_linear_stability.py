from unittest import TestCase
import numpy as np
import csdl_alpha as csdl
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import os
from unittest.mock import patch, MagicMock

from falco.core.dynamics.linear_stability import (
    LinearStabilityMetrics, 
    LinearStabilityAnalysis, 
    EigenValueOperation
)


class TestLinearStabilityMetrics(TestCase):
    """Test cases for the LinearStabilityMetrics dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        
        # Use non-interactive backend to prevent GUI popups during testing
        matplotlib.use('Agg')
    
    def test_linear_stability_metrics_init(self):
        """Test initialization of LinearStabilityMetrics."""
        a_mat_longitudinal = csdl.Variable(shape=(4, 4), value=np.eye(4), name='A_mat_L')
        real_eig_short_period = csdl.Variable(shape=(1,), value=-1.5, name='lambda_sp_real')
        imag_eig_short_period = csdl.Variable(shape=(1,), value=2.0, name='lambda_sp_imag')
        nat_freq_short_period = csdl.Variable(shape=(1,), value=2.5, name='omega_n_sp')
        damping_ratio_short_period = csdl.Variable(shape=(1,), value=0.6, name='zeta_sp')
        time_2_double_short_period = csdl.Variable(shape=(1,), value=0.5, name='t2d_sp')
        
        real_eig_phugoid = csdl.Variable(shape=(1,), value=-0.1, name='lambda_ph_real')
        imag_eig_phugoid = csdl.Variable(shape=(1,), value=0.5, name='lambda_ph_imag')
        nat_freq_phugoid = csdl.Variable(shape=(1,), value=0.51, name='omega_n_ph')
        damping_ratio_phugoid = csdl.Variable(shape=(1,), value=0.2, name='zeta_ph')
        time_2_double_phugoid = csdl.Variable(shape=(1,), value=7.0, name='t2d_ph')
        
        eig_vecs_real_longitudinal = csdl.Variable(shape=(4, 4), value=np.eye(4), name='eig_vecs_real')
        eig_vecs_imag_longitudinal = csdl.Variable(shape=(4, 4), value=np.zeros((4, 4)), name='eig_vecs_imag')
        
        # Create LinearStabilityMetrics instance
        metrics = LinearStabilityMetrics(
            A_mat_longitudinal=a_mat_longitudinal,
            real_eig_short_period=real_eig_short_period,
            imag_eig_short_period=imag_eig_short_period,
            nat_freq_short_period=nat_freq_short_period,
            damping_ratio_short_period=damping_ratio_short_period,
            time_2_double_short_period=time_2_double_short_period,
            real_eig_phugoid=real_eig_phugoid,
            imag_eig_phugoid=imag_eig_phugoid,
            nat_freq_phugoid=nat_freq_phugoid,
            damping_ratio_phugoid=damping_ratio_phugoid,
            time_2_double_phugoid=time_2_double_phugoid,
            eig_vecs_real_longitudinal=eig_vecs_real_longitudinal,
            eig_vecs_imag_longitudinal=eig_vecs_imag_longitudinal
        )
        
        # Verify all attributes are set correctly
        self.assertEqual(metrics.A_mat_longitudinal.shape, (4, 4))
        self.assertEqual(metrics.real_eig_short_period.value.item(), -1.5)
        self.assertEqual(metrics.imag_eig_short_period.value.item(), 2.0)
        self.assertEqual(metrics.nat_freq_short_period.value.item(), 2.5)
        self.assertEqual(metrics.damping_ratio_short_period.value.item(), 0.6)
        self.assertEqual(metrics.time_2_double_short_period.value.item(), 0.5)
        
        self.assertEqual(metrics.real_eig_phugoid.value.item(), -0.1)
        self.assertEqual(metrics.imag_eig_phugoid.value.item(), 0.5)
        self.assertEqual(metrics.nat_freq_phugoid.value.item(), 0.51)
        self.assertEqual(metrics.damping_ratio_phugoid.value.item(), 0.2)
        self.assertEqual(metrics.time_2_double_phugoid.value.item(), 7.0)
        
        self.assertEqual(metrics.eig_vecs_real_longitudinal.shape, (4, 4))
        self.assertEqual(metrics.eig_vecs_imag_longitudinal.shape, (4, 4))
    
    def test_linear_stability_metrics_optional_eigenvectors(self):
        """Test LinearStabilityMetrics initialization with optional eigenvectors as None."""
        # Create minimal required variables
        a_mat_longitudinal = csdl.Variable(shape=(4, 4), value=np.eye(4), name='A_mat_L')
        real_eig_short_period = csdl.Variable(shape=(1,), value=-1.5, name='lambda_sp_real')
        imag_eig_short_period = csdl.Variable(shape=(1,), value=2.0, name='lambda_sp_imag')
        nat_freq_short_period = csdl.Variable(shape=(1,), value=2.5, name='omega_n_sp')
        damping_ratio_short_period = csdl.Variable(shape=(1,), value=0.6, name='zeta_sp')
        time_2_double_short_period = csdl.Variable(shape=(1,), value=0.5, name='t2d_sp')
        
        real_eig_phugoid = csdl.Variable(shape=(1,), value=-0.1, name='lambda_ph_real')
        imag_eig_phugoid = csdl.Variable(shape=(1,), value=0.5, name='lambda_ph_imag')
        nat_freq_phugoid = csdl.Variable(shape=(1,), value=0.51, name='omega_n_ph')
        damping_ratio_phugoid = csdl.Variable(shape=(1,), value=0.2, name='zeta_ph')
        time_2_double_phugoid = csdl.Variable(shape=(1,), value=7.0, name='t2d_ph')
        
        # Create instance without optional eigenvectors
        metrics = LinearStabilityMetrics(
            A_mat_longitudinal=a_mat_longitudinal,
            real_eig_short_period=real_eig_short_period,
            imag_eig_short_period=imag_eig_short_period,
            nat_freq_short_period=nat_freq_short_period,
            damping_ratio_short_period=damping_ratio_short_period,
            time_2_double_short_period=time_2_double_short_period,
            real_eig_phugoid=real_eig_phugoid,
            imag_eig_phugoid=imag_eig_phugoid,
            nat_freq_phugoid=nat_freq_phugoid,
            damping_ratio_phugoid=damping_ratio_phugoid,
            time_2_double_phugoid=time_2_double_phugoid
        )
        
        # Verify optional attributes are None by default
        self.assertIsNone(metrics.eig_vecs_real_longitudinal)
        self.assertIsNone(metrics.eig_vecs_imag_longitudinal)


class TestEigenValueOperation(TestCase):
    """Test cases for the EigenValueOperation custom CSDL operation."""
    
    def setUp(self):
        """Set up test fixtures."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_eigenvalue_operation_init(self):
        """Test EigenValueOperation can be instantiated."""
        eig_op = EigenValueOperation()
        self.assertIsInstance(eig_op, EigenValueOperation)
    
    def test_eigenvalue_operation_simple_matrix(self):
        """Test eigenvalue computation for a simple diagonal matrix."""
        # Create a simple diagonal matrix
        test_matrix = csdl.Variable(shape=(3, 3), value=np.diag([1.0, 2.0, 3.0]), name='test_mat')
        
        eig_op = EigenValueOperation()
        eig_real, eig_imag, eig_vecs_real, eig_vecs_imag = eig_op.evaluate(test_matrix)
        
        # Check output shapes
        self.assertEqual(eig_real.shape, (3,))
        self.assertEqual(eig_imag.shape, (3,))
        self.assertEqual(eig_vecs_real.shape, (3, 3))
        self.assertEqual(eig_vecs_imag.shape, (3, 3))
        
        # For a diagonal matrix, eigenvalues should be the diagonal elements (sorted by magnitude)
        expected_eigenvals = np.array([3.0, 2.0, 1.0])  # Sorted in descending order by magnitude
        np.testing.assert_array_almost_equal(eig_real.value, expected_eigenvals, decimal=6)
        np.testing.assert_array_almost_equal(eig_imag.value, np.zeros(3), decimal=6)
    
    def test_eigenvalue_operation_complex_eigenvalues(self):
        """Test eigenvalue computation for a matrix with complex eigenvalues."""
        # Create a 2x2 rotation matrix (has complex eigenvalues)
        theta = np.pi / 4  # 45 degrees
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                   [np.sin(theta), np.cos(theta)]])
        test_matrix = csdl.Variable(shape=(2, 2), value=rotation_matrix, name='rotation_mat')
        
        eig_op = EigenValueOperation()
        eig_real, eig_imag, eig_vecs_real, eig_vecs_imag = eig_op.evaluate(test_matrix)
        
        # Check output shapes
        self.assertEqual(eig_real.shape, (2,))
        self.assertEqual(eig_imag.shape, (2,))
        self.assertEqual(eig_vecs_real.shape, (2, 2))
        self.assertEqual(eig_vecs_imag.shape, (2, 2))
        
        # For a rotation matrix, eigenvalues should be e^(±iθ) = cos(θ) ± i*sin(θ)
        expected_real = np.cos(theta)
        expected_imag = np.sin(theta)
        
        # Check that we have conjugate pair eigenvalues
        np.testing.assert_almost_equal(abs(eig_real.value[0]), expected_real, decimal=6)
        np.testing.assert_almost_equal(abs(eig_real.value[1]), expected_real, decimal=6)
        np.testing.assert_almost_equal(abs(eig_imag.value[0]), expected_imag, decimal=6)
        np.testing.assert_almost_equal(abs(eig_imag.value[1]), expected_imag, decimal=6)
    
    def test_eigenvalue_operation_4x4_matrix(self):
        """Test eigenvalue computation for a 4x4 matrix (typical for longitudinal dynamics)."""
        # Create a typical aircraft longitudinal dynamics matrix
        a_long = np.array([
            [-0.1, 0.0, 1.0, -9.81],  # u equation
            [0.0, -0.5, 10.0, 0.0],   # w equation  
            [0.0, -0.1, -2.0, 0.0],   # q equation
            [0.0, 0.0, 1.0, 0.0]      # theta equation
        ])
        test_matrix = csdl.Variable(shape=(4, 4), value=a_long, name='A_longitudinal')
        
        eig_op = EigenValueOperation()
        eig_real, eig_imag, eig_vecs_real, eig_vecs_imag = eig_op.evaluate(test_matrix)
        
        # Check output shapes
        self.assertEqual(eig_real.shape, (4,))
        self.assertEqual(eig_imag.shape, (4,))
        self.assertEqual(eig_vecs_real.shape, (4, 4))
        self.assertEqual(eig_vecs_imag.shape, (4, 4))
        
        # Check that eigenvalues are ordered by magnitude (largest first)
        magnitudes = np.sqrt(eig_real.value**2 + eig_imag.value**2)
        self.assertTrue(np.all(magnitudes[:-1] >= magnitudes[1:]))


class TestLinearStabilityAnalysis(TestCase):
    """Test cases for the LinearStabilityAnalysis class."""
    
    def setUp(self):
        """Set up test fixtures."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        
        # Use non-interactive backend
        matplotlib.use('Agg')
        
        # Create test A and B matrices typical for aircraft dynamics
        self.A_test = csdl.Variable(shape=(12, 12), value=np.zeros((12, 12)), name='A_test')
        
        # Indices for longitudinal: u(0), w(2), q(4), theta(7)
        a_long_values = np.array([
            [-0.1, 0.0, 1.0, -9.81],    # u equation
            [0.0, -0.5, 10.0, 0.0],     # w equation  
            [0.0, -0.1, -2.0, 0.0],     # q equation
            [0.0, 0.0, 1.0, 0.0]        # theta equation
        ])
        
        long_indices = [0, 2, 4, 7]
        for i, idx_i in enumerate(long_indices):
            for j, idx_j in enumerate(long_indices):
                self.A_test = self.A_test.set(csdl.slice[idx_i, idx_j], a_long_values[i, j])
        
        self.B_test = csdl.Variable(shape=(12, 4), value=np.eye(12, 4), name='B_test')
        
        self.stability_analysis = LinearStabilityAnalysis()
    
    def test_linear_stability_analysis_initialization(self):
        """Test LinearStabilityAnalysis class can be instantiated."""
        analysis = LinearStabilityAnalysis()
        self.assertIsInstance(analysis, LinearStabilityAnalysis)
    
    def test_linear_stab_analysis_none_matrices(self):
        """Test that ValueError is raised when A or B matrices are None."""
        with self.assertRaises(ValueError) as context:
            self.stability_analysis.linear_stab_analysis(None, self.B_test)
        self.assertIn("Matrices A and B must be defined", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.stability_analysis.linear_stab_analysis(self.A_test, None)
        self.assertIn("Matrices A and B must be defined", str(context.exception))
    
    def test_linear_stab_analysis_valid_matrices(self):
        """Test linear stability analysis with valid A and B matrices."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        # Check that we get a LinearStabilityMetrics object
        self.assertIsInstance(metrics, LinearStabilityMetrics)
        
        # Check that all required attributes are present and have correct shapes
        self.assertEqual(metrics.A_mat_longitudinal.shape, (4, 4))
        self.assertEqual(metrics.real_eig_short_period.shape, (1,))
        self.assertEqual(metrics.imag_eig_short_period.shape, (1,))
        self.assertEqual(metrics.nat_freq_short_period.shape, (1,))
        self.assertEqual(metrics.damping_ratio_short_period.shape, (1,))
        self.assertEqual(metrics.time_2_double_short_period.shape, (1,))
        
        self.assertEqual(metrics.real_eig_phugoid.shape, (1,))
        self.assertEqual(metrics.imag_eig_phugoid.shape, (1,))
        self.assertEqual(metrics.nat_freq_phugoid.shape, (1,))
        self.assertEqual(metrics.damping_ratio_phugoid.shape, (1,))
        self.assertEqual(metrics.time_2_double_phugoid.shape, (1,))
        
        self.assertEqual(metrics.eig_vecs_real_longitudinal.shape, (4, 4))
        self.assertEqual(metrics.eig_vecs_imag_longitudinal.shape, (4, 4))
    
    def test_longitudinal_matrix_extraction(self):
        """Test that longitudinal submatrix is correctly extracted."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        # Check that longitudinal matrix has the correct values
        a_long = metrics.A_mat_longitudinal.value
        
        # The extracted matrix should match our original longitudinal values
        expected_a_long = np.array([
            [-0.1, 0.0, 1.0, -9.81],
            [0.0, -0.5, 10.0, 0.0],
            [0.0, -0.1, -2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        np.testing.assert_array_almost_equal(a_long, expected_a_long, decimal=6)
    
    def test_mode_identification(self):
        """Test that short period and phugoid modes are correctly identified."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        # Short period mode should have higher frequency than phugoid
        sp_freq = metrics.nat_freq_short_period.value.item()
        phugoid_freq = metrics.nat_freq_phugoid.value.item()
        
        self.assertGreater(sp_freq, phugoid_freq, 
                          "Short period frequency should be higher than phugoid frequency")
    
    def test_stability_metrics_computation(self):
        """Test that stability metrics are computed correctly."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        # Test that natural frequency is computed correctly from real and imaginary parts
        sp_real = metrics.real_eig_short_period.value.item()
        sp_imag = metrics.imag_eig_short_period.value.item()
        sp_omega_n = metrics.nat_freq_short_period.value.item()
        
        expected_omega_n = np.sqrt(sp_real**2 + sp_imag**2)
        self.assertAlmostEqual(sp_omega_n, expected_omega_n, places=6)
        
        # Test that damping ratio is computed correctly
        sp_damping = metrics.damping_ratio_short_period.value.item()
        expected_damping = -sp_real / expected_omega_n
        self.assertAlmostEqual(sp_damping, expected_damping, places=6)
    
    @patch('matplotlib.pyplot.show')  # Prevent plot from displaying during test
    def test_plot_eigenvalues(self, mock_show):
        """Test eigenvalue plotting functionality."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        # Test plot generation
        fig = self.stability_analysis.plot_eigenvalues(metrics, "Test Aircraft")
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Check figure has correct structure
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        
        # Check that axes have correct labels
        ax1 = fig.axes[0]
        self.assertEqual(ax1.get_xlabel(), 'Real Part (1/s)')
        self.assertEqual(ax1.get_ylabel(), 'Imaginary Part (rad/s)')
        self.assertEqual(ax1.get_title(), 'Longitudinal Modes')
        
        plt.close(fig)  # Clean up
    
    @patch('matplotlib.pyplot.show')  # Prevent plot from displaying during test
    def test_plot_mode_characteristics(self, mock_show):
        """Test mode characteristics plotting functionality."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        # Suppress the debug print statements during test
        with patch('builtins.print'):
            fig = self.stability_analysis.plot_mode_characteristics(metrics)
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Check figure has correct structure
        self.assertEqual(len(fig.axes), 3)  # Should have 3 subplots
        
        # Check subplot titles
        self.assertEqual(fig.axes[0].get_title(), 'Natural Frequencies')
        self.assertEqual(fig.axes[1].get_title(), 'Damping Ratios')
        self.assertEqual(fig.axes[2].get_title(), 'Time to Double Amplitude')
        
        plt.close(fig)  # Clean up
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')  # Suppress print statements
    def test_generate_stability_report_no_save(self, mock_print, mock_show):
        """Test stability report generation without saving plots."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        fig1, fig2 = self.stability_analysis.generate_stability_report(
            metrics, save_plots=False
        )
        
        # Check that figures are returned
        self.assertIsInstance(fig1, plt.Figure)
        self.assertIsInstance(fig2, plt.Figure)
        
        plt.close(fig1)
        plt.close(fig2)
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')  # Suppress print statements
    def test_generate_stability_report_with_save(self, mock_print, mock_show):
        """Test stability report generation with saving plots."""
        metrics = self.stability_analysis.linear_stab_analysis(self.A_test, self.B_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            fig1, fig2 = self.stability_analysis.generate_stability_report(
                metrics, save_plots=True, plot_dir=temp_dir
            )
            
            # Check that plots were saved
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "eigenvalues.png")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "mode_characteristics.png")))
            
            plt.close(fig1)
            plt.close(fig2)
    
    def test_identify_longitudinal_modes_method(self):
        """Test the _identify_longitudinal_modes method directly."""
        # Create mock eigenvector data representing typical aircraft modes
        eig_vecs_real = csdl.Variable(shape=(4, 4), value=np.array([
            [0.1, 0.8, 0.3, 0.2],  # u components
            [0.2, 0.1, 0.9, 0.1],  # w components (should dominate short period)
            [0.1, 0.1, 0.8, 0.2],  # q components (should dominate short period)
            [0.9, 0.2, 0.1, 0.8]   # theta components (should dominate phugoid)
        ]), name='test_eig_vecs')
        
        eig_real = csdl.Variable(shape=(4,), value=np.array([-2.0, -0.1, -3.0, -0.05]), name='eig_real')
        eig_imag = csdl.Variable(shape=(4,), value=np.array([1.5, 0.3, 2.0, 0.2]), name='eig_imag')
        
        sp_idx, phugoid_idx = self.stability_analysis._identify_longitudinal_modes(
            eig_vecs_real, eig_real, eig_imag
        )
        
        # Check that we get valid indices
        self.assertIn(sp_idx, [0, 1, 2, 3])
        self.assertIn(phugoid_idx, [0, 1, 2, 3])
        self.assertNotEqual(sp_idx, phugoid_idx)
    
    def test_special_matrix_cases(self):
        """Test stability analysis with special matrix cases."""
        # Test with stable system (all eigenvalues in left half plane)
        a_stable = csdl.Variable(shape=(12, 12), value=-0.1 * np.eye(12), name='A_stable')
        a_stable = a_stable.set(csdl.slice[0, 0], -0.5)
        a_stable = a_stable.set(csdl.slice[2, 2], -1.0)
        a_stable = a_stable.set(csdl.slice[4, 4], -2.0)
        a_stable = a_stable.set(csdl.slice[7, 7], 0.0)  # Pure integrator for theta
        
        metrics = self.stability_analysis.linear_stab_analysis(a_stable, self.B_test)
        
        # All real parts should be negative or zero (stable)
        self.assertLessEqual(metrics.real_eig_short_period.value.item(), 0.0)
        self.assertLessEqual(metrics.real_eig_phugoid.value.item(), 0.0)
    
    def test_oscillatory_modes(self):
        """Test stability analysis with oscillatory modes (complex eigenvalues)."""
        # Create A matrix with complex eigenvalues
        a_osc = csdl.Variable(shape=(12, 12), value=np.zeros((12, 12)), name='A_oscillatory')
        
        # Create 2x2 oscillatory block for short period
        a_osc = a_osc.set(csdl.slice[0, 2], 1.0)
        a_osc = a_osc.set(csdl.slice[2, 0], -1.0)
        a_osc = a_osc.set(csdl.slice[0, 0], -0.5)
        a_osc = a_osc.set(csdl.slice[2, 2], -0.5)
        
        # Theta equation
        a_osc = a_osc.set(csdl.slice[7, 4], 1.0)
        
        metrics = self.stability_analysis.linear_stab_analysis(a_osc, self.B_test)
        
        # Should have non-zero imaginary parts for oscillatory modes
        self.assertNotEqual(metrics.imag_eig_short_period.value.item(), 0.0)


class TestIntegrationWithEoM(TestCase):
    """Integration tests with the EoM module."""
    
    def setUp(self):
        """Set up test fixtures."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        matplotlib.use('Agg')
    
    def test_stability_analysis_workflow(self):
        """Test complete workflow from A/B matrices to stability report."""
        # Create realistic aircraft A matrix (simplified)
        A = csdl.Variable(shape=(12, 12), value=np.zeros((12, 12)), name='A_aircraft')
        
        # Longitudinal dynamics (u, w, q, theta at indices 0, 2, 4, 7)
        long_dynamics = np.array([
            [-0.045, 0.036, 0.0, -9.81],  # u equation
            [-0.369, -0.591, 17.4, 0.0],  # w equation
            [0.0, -0.545, -1.89, 0.0],    # q equation
            [0.0, 0.0, 1.0, 0.0]          # theta equation
        ])
        
        long_indices = [0, 2, 4, 7]
        for i, idx_i in enumerate(long_indices):
            for j, idx_j in enumerate(long_indices):
                A = A.set(csdl.slice[idx_i, idx_j], long_dynamics[i, j])
        
        B = csdl.Variable(shape=(12, 4), value=np.zeros((12, 4)), name='B_aircraft')
        B = B.set(csdl.slice[4, 0], 1.0)  # Elevator control
        
        # Perform stability analysis
        analysis = LinearStabilityAnalysis()
        metrics = analysis.linear_stab_analysis(A, B)
        
        # Verify reasonable aircraft stability characteristics
        sp_freq = metrics.nat_freq_short_period.value.item()
        phugoid_freq = metrics.nat_freq_phugoid.value.item()
        
        # Typical aircraft characteristics
        self.assertGreater(sp_freq, 0.5, "Short period frequency should be reasonable")
        self.assertLess(sp_freq, 10.0, "Short period frequency should not be too high")
        self.assertGreater(phugoid_freq, 0.01, "Phugoid frequency should be positive")
        self.assertLess(phugoid_freq, 1.0, "Phugoid frequency should be low")
        
        # Short period should be higher frequency than phugoid
        self.assertGreater(sp_freq, phugoid_freq)


if __name__ == '__main__':
    import unittest
    unittest.main()
