from abc import abstractmethod
import numpy as np
import csdl_alpha as csdl
import numpy as np
from typing import Union
from dataclasses import dataclass
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem
from flight_simulator.core.dynamics.EoM import EquationsOfMotion
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.mass_properties import MassMI
import matplotlib.pyplot as plt


@dataclass
class LinearStabilityMetrics(csdl.VariableGroup):
    """Holds metrics for linear stability analysis of an aircraft.

    Attributes
    ----------
    A_mat_longitudinal : csdl.Variable
        Longitudinal state matrix.
    real_eig_short_period : csdl.Variable
        Real part of short period eigenvalue.
    imag_eig_short_period : csdl.Variable
        Imaginary part of short period eigenvalue.
    nat_freq_short_period : csdl.Variable
        Natural frequency of short period mode.
    damping_ratio_short_period : csdl.Variable
        Damping ratio of short period mode.
    time_2_double_short_period : csdl.Variable
        Time to double amplitude for short period mode.
    real_eig_phugoid : csdl.Variable
        Real part of phugoid eigenvalue.
    imag_eig_phugoid : csdl.Variable
        Imaginary part of phugoid eigenvalue.
    nat_freq_phugoid : csdl.Variable
        Natural frequency of phugoid mode.
    damping_ratio_phugoid : csdl.Variable
        Damping ratio of phugoid mode.
    time_2_double_phugoid : csdl.Variable
        Time to double amplitude for phugoid mode.
    eig_vecs_real_longitudinal : csdl.Variable, optional
        Real part of longitudinal eigenvectors.
    eig_vecs_imag_longitudinal : csdl.Variable, optional
        Imaginary part of longitudinal eigenvectors.
    """
    A_mat_longitudinal : csdl.Variable
    real_eig_short_period : csdl.Variable
    imag_eig_short_period : csdl.Variable
    nat_freq_short_period : csdl.Variable
    damping_ratio_short_period : csdl.Variable
    time_2_double_short_period : csdl.Variable

    real_eig_phugoid : csdl.Variable
    imag_eig_phugoid : csdl.Variable
    nat_freq_phugoid : csdl.Variable
    damping_ratio_phugoid : csdl.Variable
    time_2_double_phugoid : csdl.Variable

    # A_mat_lateral_directional : csdl.Variable
    # real_eig_spiral : csdl.Variable
    # imag_eig_spiral : csdl.Variable
    # nat_freq_spiral : csdl.Variable
    # damping_ratio_spiral : csdl.Variable
    # time_2_double_spiral : csdl.Variable

    # real_eig_dutch_roll : csdl.Variable
    # imag_eig_dutch_roll : csdl.Variable
    # nat_freq_dutch_roll : csdl.Variable
    # damping_ratio_dutch_roll : csdl.Variable
    # time_2_double_dutch_roll : csdl.Variable
    
    # real_eig_roll : csdl.Variable
    # imag_eig_roll : csdl.Variable
    # nat_freq_roll : csdl.Variable
    # damping_ratio_roll : csdl.Variable
    # time_2_double_roll : csdl.Variable
    
    eig_vecs_real_longitudinal : csdl.Variable = None
    eig_vecs_imag_longitudinal : csdl.Variable = None
    # eig_vecs_real_lateral : csdl.Variable = None
    # eig_vecs_imag_lateral : csdl.Variable = None

class LinearStabilityAnalysis():
    """Performs linear stability analysis for aircraft dynamic modes.

    Provides methods to compute eigenvalues, mode characteristics, and generate plots/reports.
    """

    def linear_stab_analysis(self, A, B):
        """Perform longitudinal linear stability analysis on a specified aircraft state.

        Parameters
        ----------
        A : csdl.Variable or np.ndarray
            State matrix (system dynamics).
        B : csdl.Variable or np.ndarray
            Input matrix (control dynamics).

        Returns
        -------
        LinearStabilityMetrics
            Object containing computed stability metrics.
        """

        # Ensure A and B are defined
        if A is None or B is None:
            raise ValueError("Matrices A and B must be defined for linear stability analysis.")
        
        # Long: u, w, q, theta
        long_indices=[0, 2, 4, 7]
        A_mat_L = csdl.Variable(value=0, shape=(4, 4), name="A_mat_Longitudinal")
        # Extract the 4x4 longitudinal submatrix
        for i in range(4):
            for j in range(4):
                A_mat_L = A_mat_L.set(csdl.slice[i, j], A[long_indices[i], long_indices[j]])

        
        
        eig_val_long_operation = EigenValueOperation()
        eig_real_long, eig_imag_long, eig_vecs_real_long, eig_vecs_imag_long = eig_val_long_operation.evaluate(A_mat_L)
        
        
        # Identify longitudinal modes based on eigenvector characteristics
        # State order for longitudinal: [u, w, q, theta] (indices 0, 1, 2, 3)
        sp_idx, phugoid_idx = self._identify_longitudinal_modes(eig_vecs_real_long, eig_real_long, eig_imag_long)
        
        # Short period mode
        lambda_sp_real = eig_real_long[sp_idx]
        lambda_sp_imag = eig_imag_long[sp_idx]
        sp_omega_n = ((lambda_sp_real ** 2 + lambda_sp_imag ** 2) + 1e-10) ** 0.5
        sp_damping_ratio = -lambda_sp_real / sp_omega_n
        sp_time_2_double = csdl.log(2) / ((lambda_sp_real ** 2 + 1e-10) ** 0.5)

        # Phugoid mode
        lambda_phugoid_real = eig_real_long[phugoid_idx]
        lambda_phugoid_imag = eig_imag_long[phugoid_idx]
        phugoid_omega_n = ((lambda_phugoid_real ** 2 + lambda_phugoid_imag ** 2) + 1e-10) ** 0.5
        phugoid_damping_ratio = -lambda_phugoid_real / phugoid_omega_n
        phugoid_time_2_double = csdl.log(2) / ((lambda_phugoid_real ** 2 + 1e-10) ** 0.5)


        # ## LATERAL-DIRECTIONAL MODES
        # # Lat-Dir: v, p, r, phi
        # A_mat_LD = csdl.Variable(value=0, shape=(4, 4), name="A_mat_Lateral_Directional")
        # latdir_indices = [1, 3, 5, 6]  # v, p, r, phi

        # # Extract the 4x4 lateral-directional submatrix
        # for i in range(4):
        #     for j in range(4):
        #         A_mat_LD = A_mat_LD.set(csdl.slice[i, j], A[latdir_indices[i], latdir_indices[j]])

        # eig_val_lat_operation = EigenValueOperation()
        # eig_real_lat, eig_imag_lat, eig_vecs_real_lat, eig_vecs_imag_lat = eig_val_lat_operation.evaluate(A_mat_LD)

        # # Identify lateral-directional modes based on eigenvector characteristics  
        # # State order for lateral: [v, p, r, phi] (indices 0, 1, 2, 3)
        # spiral_idx, dutch_roll_idx, roll_idx = self._identify_lateral_modes(eig_vecs_real_lat, eig_real_lat, eig_imag_lat)
        
        # # Spiral mode
        # lambda_spiral_real = eig_real_lat[spiral_idx]
        # lambda_spiral_imag = eig_imag_lat[spiral_idx]
        # spiral_omega_n = ((lambda_spiral_real ** 2 + lambda_spiral_imag ** 2) + 1e-10) ** 0.5
        # spiral_damping_ratio = -lambda_spiral_real / spiral_omega_n
        # spiral_time_2_double = csdl.log(2) / ((lambda_spiral_real ** 2 + 1e-10) ** 0.5)

        # # Dutch Roll mode
        # lambda_dutch_roll_real = eig_real_lat[dutch_roll_idx]
        # lambda_dutch_roll_imag = eig_imag_lat[dutch_roll_idx]
        # dutch_roll_omega_n = ((lambda_dutch_roll_real ** 2 + lambda_dutch_roll_imag ** 2) + 1e-10) ** 0.5
        # dutch_roll_damping_ratio = -lambda_dutch_roll_real / dutch_roll_omega_n
        # dutch_roll_time_2_double = csdl.log(2) / ((lambda_dutch_roll_real ** 2 + 1e-10) ** 0.5)

        # # Roll Subsidence mode
        # lambda_roll_real = eig_real_lat[roll_idx]
        # lambda_roll_imag = eig_imag_lat[roll_idx]
        # roll_omega_n = ((lambda_roll_real ** 2 + lambda_roll_imag ** 2) + 1e-10) ** 0.5
        # roll_damping_ratio = -lambda_roll_real / roll_omega_n
        # roll_time_2_double = csdl.log(2) / ((lambda_roll_real ** 2 + 1e-10) ** 0.5)

        stability_analysis = LinearStabilityMetrics(
            A_mat_longitudinal=A_mat_L,
            real_eig_short_period=lambda_sp_real,
            imag_eig_short_period=lambda_sp_imag,
            nat_freq_short_period=sp_omega_n,
            damping_ratio_short_period=sp_damping_ratio,
            time_2_double_short_period=sp_time_2_double,
            real_eig_phugoid=lambda_phugoid_real,
            imag_eig_phugoid=lambda_phugoid_imag,
            nat_freq_phugoid=phugoid_omega_n,
            damping_ratio_phugoid=phugoid_damping_ratio,
            time_2_double_phugoid=phugoid_time_2_double,
            # A_mat_lateral_directional=A_mat_LD,
            # real_eig_spiral=lambda_spiral_real,
            # imag_eig_spiral=lambda_spiral_imag,
            # nat_freq_spiral=spiral_omega_n,
            # damping_ratio_spiral=spiral_damping_ratio,
            # time_2_double_spiral=spiral_time_2_double,
            # real_eig_dutch_roll=lambda_dutch_roll_real,
            # imag_eig_dutch_roll=lambda_dutch_roll_imag,
            # nat_freq_dutch_roll=dutch_roll_omega_n,
            # damping_ratio_dutch_roll=dutch_roll_damping_ratio,
            # time_2_double_dutch_roll=dutch_roll_time_2_double,
            # real_eig_roll=lambda_roll_real,
            # imag_eig_roll=lambda_roll_imag,
            # nat_freq_roll=roll_omega_n,
            # damping_ratio_roll=roll_damping_ratio,
            # time_2_double_roll=roll_time_2_double,
            eig_vecs_real_longitudinal=eig_vecs_real_long,
            eig_vecs_imag_longitudinal=eig_vecs_imag_long,
            # eig_vecs_real_lateral=eig_vecs_real_lat,
            # eig_vecs_imag_lateral=eig_vecs_imag_lat
        )
        
        return stability_analysis
    
    def plot_eigenvalues(self, stability_metrics: LinearStabilityMetrics, 
                    title: str = "Aircraft Stability Analysis"):
        
        """Plot eigenvalues on the complex plane (s-plane).

        Parameters
        ----------
        stability_metrics : LinearStabilityMetrics
            Stability metrics containing eigenvalues.
        title : str, optional
            Title for the plot.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object for the eigenvalue plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Longitudinal modes
        ax1.scatter(stability_metrics.real_eig_short_period.value, 
                   stability_metrics.imag_eig_short_period.value, 
                   color='red', s=100, marker='x', label='Short Period')
        ax1.scatter(stability_metrics.real_eig_phugoid.value, 
                   stability_metrics.imag_eig_phugoid.value, 
                   color='blue', s=100, marker='o', label='Phugoid')
        
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Real Part (1/s)')
        ax1.set_ylabel('Imaginary Part (rad/s)')
        ax1.set_title('Longitudinal Modes')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # # Lateral-Directional modes
        # ax2.scatter(stability_metrics.real_eig_spiral.value, 
        #            stability_metrics.imag_eig_spiral.value, 
        #            color='green', s=100, marker='s', label='Spiral')
        # ax2.scatter(stability_metrics.real_eig_dutch_roll.value, 
        #            stability_metrics.imag_eig_dutch_roll.value, 
        #            color='orange', s=100, marker='^', label='Dutch Roll')
        # ax2.scatter(stability_metrics.real_eig_roll.value, 
        #            stability_metrics.imag_eig_roll.value, 
        #            color='purple', s=100, marker='d', label='Roll Subsidence')
        
        # ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        # ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # ax2.set_xlabel('Real Part (1/s)')
        # ax2.set_ylabel('Imaginary Part (rad/s)')
        # ax2.set_title('Lateral-Directional Modes')
        # ax2.grid(True, alpha=0.3)
        # ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_mode_characteristics(self, stability_metrics: LinearStabilityMetrics):
        """Plot natural frequency, damping ratio, and time to double for all modes.

        Parameters
        ----------
        stability_metrics : LinearStabilityMetrics
            Stability metrics containing mode characteristics.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object for the mode characteristics plot.
        """
        modes = ['Short Period', 'Phugoid'] 
                #  'Spiral', 'Dutch Roll', 'Roll']
        
        # Extract .value from each CSDL Variable to get numpy scalars
        nat_freqs = [
            stability_metrics.nat_freq_short_period.value.item() if hasattr(stability_metrics.nat_freq_short_period.value, 'item') else stability_metrics.nat_freq_short_period.value,
            stability_metrics.nat_freq_phugoid.value.item() if hasattr(stability_metrics.nat_freq_phugoid.value, 'item') else stability_metrics.nat_freq_phugoid.value,
            # stability_metrics.nat_freq_spiral.value.item() if hasattr(stability_metrics.nat_freq_spiral.value, 'item') else stability_metrics.nat_freq_spiral.value,
            # stability_metrics.nat_freq_dutch_roll.value.item() if hasattr(stability_metrics.nat_freq_dutch_roll.value, 'item') else stability_metrics.nat_freq_dutch_roll.value,
            # stability_metrics.nat_freq_roll.value.item() if hasattr(stability_metrics.nat_freq_roll.value, 'item') else stability_metrics.nat_freq_roll.value
        ]
        
        damping_ratios = [
            stability_metrics.damping_ratio_short_period.value.item() if hasattr(stability_metrics.damping_ratio_short_period.value, 'item') else stability_metrics.damping_ratio_short_period.value,
            stability_metrics.damping_ratio_phugoid.value.item() if hasattr(stability_metrics.damping_ratio_phugoid.value, 'item') else stability_metrics.damping_ratio_phugoid.value,
            # stability_metrics.damping_ratio_spiral.value.item() if hasattr(stability_metrics.damping_ratio_spiral.value, 'item') else stability_metrics.damping_ratio_spiral.value,
            # stability_metrics.damping_ratio_dutch_roll.value.item() if hasattr(stability_metrics.damping_ratio_dutch_roll.value, 'item') else stability_metrics.damping_ratio_dutch_roll.value,
            # stability_metrics.damping_ratio_roll.value.item() if hasattr(stability_metrics.damping_ratio_roll.value, 'item') else stability_metrics.damping_ratio_roll.value
        ]
        
        time_to_double = [
            stability_metrics.time_2_double_short_period.value.item() if hasattr(stability_metrics.time_2_double_short_period.value, 'item') else stability_metrics.time_2_double_short_period.value,
            stability_metrics.time_2_double_phugoid.value.item() if hasattr(stability_metrics.time_2_double_phugoid.value, 'item') else stability_metrics.time_2_double_phugoid.value,
            # stability_metrics.time_2_double_spiral.value.item() if hasattr(stability_metrics.time_2_double_spiral.value, 'item') else stability_metrics.time_2_double_spiral.value,
            # stability_metrics.time_2_double_dutch_roll.value.item() if hasattr(stability_metrics.time_2_double_dutch_roll.value, 'item') else stability_metrics.time_2_double_dutch_roll.value,
            # stability_metrics.time_2_double_roll.value.item() if hasattr(stability_metrics.time_2_double_roll.value, 'item') else stability_metrics.time_2_double_roll.value
        ]
        
        # Debug print to check the values
        print("Natural frequencies:", nat_freqs)
        print("Damping ratios:", damping_ratios)
        print("Time to double:", time_to_double)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Natural Frequency
        bars1 = ax1.bar(modes, nat_freqs, color=['red', 'blue']) # 'green', 'orange', 'purple'])
        ax1.set_ylabel('Natural Frequency (rad/s)')
        ax1.set_title('Natural Frequencies')
        ax1.tick_params(axis='x', rotation=45)
        
        # Damping Ratio
        bars2 = ax2.bar(modes, damping_ratios, color=['red', 'blue']) #, 'green', 'orange', 'purple'])
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Critical Damping')
        ax2.set_ylabel('Damping Ratio')
        ax2.set_title('Damping Ratios')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Time to Double
        bars3 = ax3.bar(modes, time_to_double, color=['red', 'blue'])#, 'green', 'orange', 'purple'])
        ax3.set_ylabel('Time to Double (s)')
        ax3.set_title('Time to Double Amplitude')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    
    def generate_stability_report(self, stability_metrics: LinearStabilityMetrics, 
                                save_plots: bool = True, plot_dir: str = "./plots/"):
        """Generate a comprehensive stability analysis report with plots.

        Parameters
        ----------
        stability_metrics : LinearStabilityMetrics
            Stability metrics to report.
        save_plots : bool, optional
            Whether to save plots to disk (default True).
        plot_dir : str, optional
            Directory to save plots (default "./plots/").

        Returns
        -------
        tuple
            Tuple of matplotlib.figure.Figure objects for the generated plots.
        """
        import os
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
        
        # Generate all plots
        fig1 = self.plot_eigenvalues(stability_metrics)
        fig2 = self.plot_mode_characteristics(stability_metrics)
        
        if save_plots:
            fig1.savefig(f"{plot_dir}/eigenvalues.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{plot_dir}/mode_characteristics.png", dpi=300, bbox_inches='tight')
        
        # Print numerical summary
        print("=" * 60)
        print("LINEAR STABILITY ANALYSIS REPORT")
        print("=" * 60)
        
        print("\nLONGITUDINAL MODES:")
        print(f"Short Period: ωn = {stability_metrics.nat_freq_short_period.value} rad/s, "
              f"ζ = {stability_metrics.damping_ratio_short_period.value}")
        print(f"Phugoid: ωn = {stability_metrics.nat_freq_phugoid.value} rad/s, "
              f"ζ = {stability_metrics.damping_ratio_phugoid.value}")
        
        # print("\nLATERAL-DIRECTIONAL MODES:")
        # print(f"Spiral: ωn = {stability_metrics.nat_freq_spiral.value} rad/s, "
        #       f"ζ = {stability_metrics.damping_ratio_spiral.value}")
        # print(f"Dutch Roll: ωn = {stability_metrics.nat_freq_dutch_roll.value} rad/s, "
        #       f"ζ = {stability_metrics.damping_ratio_dutch_roll.value}")
        # print(f"Roll Subsidence: ωn = {stability_metrics.nat_freq_roll.value} rad/s, "
        #         f"ζ = {stability_metrics.damping_ratio_roll.value}")
        
        
        plt.show()
        
        return fig1, fig2
    
    def _identify_longitudinal_modes(self, eig_vecs_real, eig_real, eig_imag):
        """
        Identify short period and phugoid modes based on eigenvector characteristics.
        
        Short period: dominated by w (angle of attack) and q (pitch rate) - indices 1, 2
        Phugoid: dominated by u (forward velocity) and theta (pitch angle) - indices 0, 3
        """
        mode_scores = csdl.Variable(value=0, shape=(4,), name="mode_classification_scores")
        
        for i in range(4):  # 4 eigenvalues/eigenvectors
            # Calculate short period score: contribution from w and q components
            w_component = csdl.absolute(eig_vecs_real[1, i])  # w component (index 1)
            q_component = csdl.absolute(eig_vecs_real[2, i])  # q component (index 2)
            sp_score = w_component + q_component
            
            # Calculate phugoid score: contribution from u and theta components  
            u_component = csdl.absolute(eig_vecs_real[0, i])    # u component (index 0)
            theta_component = csdl.absolute(eig_vecs_real[3, i]) # theta component (index 3)
            phugoid_score = u_component + theta_component
            
            # Short period typically has higher frequency than phugoid
            # Use frequency as additional criterion
            freq = ((eig_real[i] ** 2 + eig_imag[i] ** 2) + 1e-10) ** 0.5
            
            # Combined score favoring short period characteristics
            mode_scores = mode_scores.set(csdl.slice[i], sp_score + 0.1 * freq)
        
        # Find indices of max scores for classification
        sp_idx = np.argmax(mode_scores.value)
        
        # For phugoid, find the mode with highest u+theta contribution excluding short period
        phugoid_scores = csdl.Variable(value=0, shape=(4,), name="phugoid_scores")
        for i in range(4):
            u_component = csdl.absolute(eig_vecs_real[0, i])
            theta_component = csdl.absolute(eig_vecs_real[3, i])
            phugoid_scores = phugoid_scores.set(csdl.slice[i], u_component + theta_component)
        
        # Set short period score to zero to exclude it
        phugoid_scores = phugoid_scores.set(csdl.slice[sp_idx], 0.0)
        phugoid_idx = np.argmax(phugoid_scores.value)
        
        return sp_idx, phugoid_idx



    

class EigenValueOperation(csdl.CustomExplicitOperation):
    """Custom CSDL operation to compute eigenvalues and eigenvectors of a matrix.

    Provides both values and derivatives for use in CSDL computational graphs.
    """
    def __init__(self):
        super().__init__()

    def evaluate(self, mat):
        """Evaluate eigenvalues and eigenvectors for the given matrix.

        Parameters
        ----------
        mat : np.ndarray or csdl.Variable
            Square matrix to analyze.

        Returns
        -------
        tuple
            Real and imaginary parts of eigenvalues and eigenvectors.
        """
        shape = mat.shape
        size = shape[0]

        self.declare_input("mat", mat)
        eig_real = self.create_output("eig_vals_real", shape=(size, ))
        eig_imag = self.create_output("eig_vals_imag", shape=(size, ))
        eig_vecs_real = self.create_output("eig_vecs_real", shape=(size, size))
        eig_vecs_imag = self.create_output("eig_vecs_imag", shape=(size, size))

        self.declare_derivative_parameters("eig_vals_real", "mat")
        self.declare_derivative_parameters("eig_vals_imag", "mat")
        self.declare_derivative_parameters("eig_vecs_real", "mat")
        self.declare_derivative_parameters("eig_vecs_imag", "mat")

        return eig_real, eig_imag, eig_vecs_real, eig_vecs_imag

    def compute(self, inputs, outputs):
        """Compute eigenvalues and eigenvectors for the input matrix.

        Parameters
        ----------
        inputs : dict
            Dictionary of input variables.
        outputs : dict
            Dictionary to store output variables.
        """
        mat = inputs["mat"]

        eig_vals, eig_vecs = np.linalg.eig(mat)
    
        idx = np.abs(eig_vals).argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        outputs['eig_vals_real'] = np.real(eig_vals)
        outputs['eig_vals_imag'] = np.imag(eig_vals)
        outputs['eig_vecs_real'] = np.real(eig_vecs)
        outputs['eig_vecs_imag'] = np.imag(eig_vecs)

    def compute_derivatives(self, inputs, outputs, derivatives):
        """Compute derivatives of eigenvalues and eigenvectors with respect to the matrix.

        Parameters
        ----------
        inputs : dict
            Dictionary of input variables.
        outputs : dict
            Dictionary of output variables.
        derivatives : dict
            Dictionary to store computed derivatives.
        """
        mat = inputs["mat"]
        size = mat.shape[0]
        eig_vals, eig_vecs = np.linalg.eig(mat)
        idx = np.abs(eig_vals).argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        # v inverse transpose
        v_inv_T = (np.linalg.inv(eig_vecs)).T

        # preallocate Jacobian: n outputs, n^2 inputs
        temp_vals_r = np.zeros((size, size*size))
        temp_vals_i = np.zeros((size, size*size))
        temp_vecs_r = np.zeros((size*size, size*size))
        temp_vecs_i = np.zeros((size*size, size*size))

        for j in range(len(eig_vals)):
            # Eigenvalue derivatives (same as before)
            partial = np.outer(eig_vecs[:, j], v_inv_T[:, j]).flatten(order='F')
            temp_vals_r[j, :] = np.real(partial)
            temp_vals_i[j, :] = np.imag(partial)

            # Eigenvector derivatives
            for k in range(size):
                if k != j:
                    # For eigenvector derivatives, we need to compute dv_j/dA
                    # This involves the formula: dv_j/dA = sum_{k!=j} (v_k * v_j^T) / (lambda_j - lambda_k)
                    factor = 1.0 / (eig_vals[j] - eig_vals[k] + 1e-12)  # Small regularization
                    partial_vec = np.outer(eig_vecs[:, k], v_inv_T[:, j]) * factor
                    
                    # Flatten and add to derivative matrix
                    for i in range(size):
                        row_idx = j * size + i
                        temp_vecs_r[row_idx, :] += np.real(partial_vec[i, :].flatten(order='F'))
                        temp_vecs_i[row_idx, :] += np.imag(partial_vec[i, :].flatten(order='F'))

        # Set Jacobians
        derivatives['eig_vals_real', 'mat'] = temp_vals_r
        derivatives['eig_vals_imag', 'mat'] = temp_vals_i
        derivatives['eig_vecs_real', 'mat'] = temp_vecs_r
        derivatives['eig_vecs_imag', 'mat'] = temp_vecs_i
