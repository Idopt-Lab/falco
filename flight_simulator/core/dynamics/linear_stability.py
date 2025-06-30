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

    A_mat_lateral_directional : csdl.Variable
    real_eig_spiral : csdl.Variable
    imag_eig_spiral : csdl.Variable
    nat_freq_spiral : csdl.Variable
    damping_ratio_spiral : csdl.Variable
    time_2_double_spiral : csdl.Variable

    real_eig_dutch_roll : csdl.Variable
    imag_eig_dutch_roll : csdl.Variable
    nat_freq_dutch_roll : csdl.Variable
    damping_ratio_dutch_roll : csdl.Variable
    time_2_double_dutch_roll : csdl.Variable
    
    # Add roll subsidence mode
    real_eig_roll : csdl.Variable
    imag_eig_roll : csdl.Variable
    nat_freq_roll : csdl.Variable
    damping_ratio_roll : csdl.Variable
    time_2_double_roll : csdl.Variable
    
    eig_vecs_real_longitudinal : csdl.Variable = None
    eig_vecs_imag_longitudinal : csdl.Variable = None
    eig_vecs_real_lateral : csdl.Variable = None
    eig_vecs_imag_lateral : csdl.Variable = None

class LinearStabilityAnalysis():

    def linear_stab_analysis(self, A, B):
        """
        Perform longitudinal linear stability analysis on a specifed aircraft state
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

        
        # Lat-Dir: v, p, r, phi
        A_mat_LD = csdl.Variable(value=0, shape=(4, 4), name="A_mat_Lateral_Directional")
        latdir_indices = [1, 3, 5, 6]  # v, p, r, phi

        # Extract the 4x4 lateral-directional submatrix
        for i in range(4):
            for j in range(4):
                A_mat_LD = A_mat_LD.set(csdl.slice[i, j], A[latdir_indices[i], latdir_indices[j]])
    
        
        eig_val_long_operation = EigenValueOperation()
        eig_real_long, eig_imag_long, eig_vecs_real_long, eig_vecs_imag_long = eig_val_long_operation.evaluate(A_mat_L)
        
        eig_val_lat_operation = EigenValueOperation()
        eig_real_lat, eig_imag_lat, eig_vecs_real_lat, eig_vecs_imag_lat = eig_val_lat_operation.evaluate(A_mat_LD)
        
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

        # Identify lateral-directional modes based on eigenvector characteristics  
        # State order for lateral: [v, p, r, phi] (indices 0, 1, 2, 3)
        spiral_idx, dutch_roll_idx, roll_idx = self._identify_lateral_modes(eig_vecs_real_lat, eig_real_lat, eig_imag_lat)
        
        # Spiral mode
        lambda_spiral_real = eig_real_lat[spiral_idx]
        lambda_spiral_imag = eig_imag_lat[spiral_idx]
        spiral_omega_n = ((lambda_spiral_real ** 2 + lambda_spiral_imag ** 2) + 1e-10) ** 0.5
        spiral_damping_ratio = -lambda_spiral_real / spiral_omega_n
        spiral_time_2_double = csdl.log(2) / ((lambda_spiral_real ** 2 + 1e-10) ** 0.5)

        # Dutch Roll mode
        lambda_dutch_roll_real = eig_real_lat[dutch_roll_idx]
        lambda_dutch_roll_imag = eig_imag_lat[dutch_roll_idx]
        dutch_roll_omega_n = ((lambda_dutch_roll_real ** 2 + lambda_dutch_roll_imag ** 2) + 1e-10) ** 0.5
        dutch_roll_damping_ratio = -lambda_dutch_roll_real / dutch_roll_omega_n
        dutch_roll_time_2_double = csdl.log(2) / ((lambda_dutch_roll_real ** 2 + 1e-10) ** 0.5)

        # Roll Subsidence mode
        lambda_roll_real = eig_real_lat[roll_idx]
        lambda_roll_imag = eig_imag_lat[roll_idx]
        roll_omega_n = ((lambda_roll_real ** 2 + lambda_roll_imag ** 2) + 1e-10) ** 0.5
        roll_damping_ratio = -lambda_roll_real / roll_omega_n
        roll_time_2_double = csdl.log(2) / ((lambda_roll_real ** 2 + 1e-10) ** 0.5)

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
            A_mat_lateral_directional=A_mat_LD,
            real_eig_spiral=lambda_spiral_real,
            imag_eig_spiral=lambda_spiral_imag,
            nat_freq_spiral=spiral_omega_n,
            damping_ratio_spiral=spiral_damping_ratio,
            time_2_double_spiral=spiral_time_2_double,
            real_eig_dutch_roll=lambda_dutch_roll_real,
            imag_eig_dutch_roll=lambda_dutch_roll_imag,
            nat_freq_dutch_roll=dutch_roll_omega_n,
            damping_ratio_dutch_roll=dutch_roll_damping_ratio,
            time_2_double_dutch_roll=dutch_roll_time_2_double,
            real_eig_roll=lambda_roll_real,
            imag_eig_roll=lambda_roll_imag,
            nat_freq_roll=roll_omega_n,
            damping_ratio_roll=roll_damping_ratio,
            time_2_double_roll=roll_time_2_double,
            eig_vecs_real_longitudinal=eig_vecs_real_long,
            eig_vecs_imag_longitudinal=eig_vecs_imag_long,
            eig_vecs_real_lateral=eig_vecs_real_lat,
            eig_vecs_imag_lateral=eig_vecs_imag_lat
        )
        
        return stability_analysis
    
    def plot_eigenvalues(self, stability_metrics: LinearStabilityMetrics, 
                    title: str = "Aircraft Stability Analysis"):
        """Plot eigenvalues on the complex plane (s-plane)."""
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
        
        # Lateral-Directional modes
        ax2.scatter(stability_metrics.real_eig_spiral.value, 
                   stability_metrics.imag_eig_spiral.value, 
                   color='green', s=100, marker='s', label='Spiral')
        ax2.scatter(stability_metrics.real_eig_dutch_roll.value, 
                   stability_metrics.imag_eig_dutch_roll.value, 
                   color='orange', s=100, marker='^', label='Dutch Roll')
        ax2.scatter(stability_metrics.real_eig_roll.value, 
                   stability_metrics.imag_eig_roll.value, 
                   color='purple', s=100, marker='d', label='Roll Subsidence')
        
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Real Part (1/s)')
        ax2.set_ylabel('Imaginary Part (rad/s)')
        ax2.set_title('Lateral-Directional Modes')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_mode_characteristics(self, stability_metrics: LinearStabilityMetrics):
        """Plot natural frequency, damping ratio, and time to double for all modes."""
        modes = ['Short Period', 'Phugoid', 'Spiral', 'Dutch Roll', 'Roll']
        
        # Extract .value from each CSDL Variable to get numpy scalars
        nat_freqs = [
            stability_metrics.nat_freq_short_period.value.item() if hasattr(stability_metrics.nat_freq_short_period.value, 'item') else stability_metrics.nat_freq_short_period.value,
            stability_metrics.nat_freq_phugoid.value.item() if hasattr(stability_metrics.nat_freq_phugoid.value, 'item') else stability_metrics.nat_freq_phugoid.value,
            stability_metrics.nat_freq_spiral.value.item() if hasattr(stability_metrics.nat_freq_spiral.value, 'item') else stability_metrics.nat_freq_spiral.value,
            stability_metrics.nat_freq_dutch_roll.value.item() if hasattr(stability_metrics.nat_freq_dutch_roll.value, 'item') else stability_metrics.nat_freq_dutch_roll.value,
            stability_metrics.nat_freq_roll.value.item() if hasattr(stability_metrics.nat_freq_roll.value, 'item') else stability_metrics.nat_freq_roll.value
        ]
        
        damping_ratios = [
            stability_metrics.damping_ratio_short_period.value.item() if hasattr(stability_metrics.damping_ratio_short_period.value, 'item') else stability_metrics.damping_ratio_short_period.value,
            stability_metrics.damping_ratio_phugoid.value.item() if hasattr(stability_metrics.damping_ratio_phugoid.value, 'item') else stability_metrics.damping_ratio_phugoid.value,
            stability_metrics.damping_ratio_spiral.value.item() if hasattr(stability_metrics.damping_ratio_spiral.value, 'item') else stability_metrics.damping_ratio_spiral.value,
            stability_metrics.damping_ratio_dutch_roll.value.item() if hasattr(stability_metrics.damping_ratio_dutch_roll.value, 'item') else stability_metrics.damping_ratio_dutch_roll.value,
            stability_metrics.damping_ratio_roll.value.item() if hasattr(stability_metrics.damping_ratio_roll.value, 'item') else stability_metrics.damping_ratio_roll.value
        ]
        
        time_to_double = [
            stability_metrics.time_2_double_short_period.value.item() if hasattr(stability_metrics.time_2_double_short_period.value, 'item') else stability_metrics.time_2_double_short_period.value,
            stability_metrics.time_2_double_phugoid.value.item() if hasattr(stability_metrics.time_2_double_phugoid.value, 'item') else stability_metrics.time_2_double_phugoid.value,
            stability_metrics.time_2_double_spiral.value.item() if hasattr(stability_metrics.time_2_double_spiral.value, 'item') else stability_metrics.time_2_double_spiral.value,
            stability_metrics.time_2_double_dutch_roll.value.item() if hasattr(stability_metrics.time_2_double_dutch_roll.value, 'item') else stability_metrics.time_2_double_dutch_roll.value,
            stability_metrics.time_2_double_roll.value.item() if hasattr(stability_metrics.time_2_double_roll.value, 'item') else stability_metrics.time_2_double_roll.value
        ]
        
        # Debug print to check the values
        print("Natural frequencies:", nat_freqs)
        print("Damping ratios:", damping_ratios)
        print("Time to double:", time_to_double)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Natural Frequency
        bars1 = ax1.bar(modes, nat_freqs, color=['red', 'blue', 'green', 'orange', 'purple'])
        ax1.set_ylabel('Natural Frequency (rad/s)')
        ax1.set_title('Natural Frequencies')
        ax1.tick_params(axis='x', rotation=45)
        
        # Damping Ratio
        bars2 = ax2.bar(modes, damping_ratios, color=['red', 'blue', 'green', 'orange', 'purple'])
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Critical Damping')
        ax2.set_ylabel('Damping Ratio')
        ax2.set_title('Damping Ratios')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Time to Double
        bars3 = ax3.bar(modes, time_to_double, color=['red', 'blue', 'green', 'orange', 'purple'])
        ax3.set_ylabel('Time to Double (s)')
        ax3.set_title('Time to Double Amplitude')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_time_response(self, stability_metrics: LinearStabilityMetrics, 
                          t_end: float = 180.0, dt: float = 0.01):
        """Plot approximate time responses for each mode."""
        t = np.arange(0, t_end, dt)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
        ax4, ax5 = axes[1, 0], axes[1, 1]
        
        # Short Period Response
        sigma_sp = stability_metrics.real_eig_short_period.value
        omega_sp = stability_metrics.imag_eig_short_period.value
        if abs(omega_sp) > 1e-6:  # Oscillatory
            response_sp = np.exp(sigma_sp * t) * np.cos(omega_sp * t)
        else:  # Non-oscillatory
            response_sp = np.exp(sigma_sp * t)
        ax1.plot(t, response_sp, 'r-', label='Short Period')
        ax1.set_title('Short Period Mode')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Response')
        ax1.grid(True, alpha=0.3)
        
        # Phugoid Response
        sigma_ph = stability_metrics.real_eig_phugoid.value
        omega_ph = stability_metrics.imag_eig_phugoid.value
        if abs(omega_ph) > 1e-6:
            response_ph = np.exp(sigma_ph * t) * np.cos(omega_ph * t)
        else:
            response_ph = np.exp(sigma_ph * t)
        ax2.plot(t, response_ph, 'b-', label='Phugoid')
        ax2.set_title('Phugoid Mode')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Response')
        ax2.grid(True, alpha=0.3)
        
        # Spiral Response
        sigma_spiral = stability_metrics.real_eig_spiral.value
        omega_spiral = stability_metrics.imag_eig_spiral.value
        if abs(omega_spiral) > 1e-6:
            response_spiral = np.exp(sigma_spiral * t) * np.cos(omega_spiral * t)
        else:
            response_spiral = np.exp(sigma_spiral * t)
        ax3.plot(t, response_spiral, 'g-', label='Spiral')
        ax3.set_title('Spiral Mode')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Response')
        ax3.grid(True, alpha=0.3)
        
        # Dutch Roll Response
        sigma_dr = stability_metrics.real_eig_dutch_roll.value
        omega_dr = stability_metrics.imag_eig_dutch_roll.value
        if abs(omega_dr) > 1e-6:
            response_dr = np.exp(sigma_dr * t) * np.cos(omega_dr * t)
        else:
            response_dr = np.exp(sigma_dr * t)
        ax4.plot(t, response_dr, 'orange', label='Dutch Roll')
        ax4.set_title('Dutch Roll Mode')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Response')
        ax4.grid(True, alpha=0.3)

        # Roll Subsidence Response
        sigma_roll = stability_metrics.real_eig_roll.value
        omega_roll = stability_metrics.imag_eig_roll.value
        if abs(omega_roll) > 1e-6:
            response_roll = np.exp(sigma_roll * t) * np.cos(omega_roll * t)
        else:
            response_roll = np.exp(sigma_roll * t)
        ax5.plot(t, response_roll, 'purple', label='Roll Subsidence')
        ax5.set_title('Roll Subsidence Mode')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Response')
        ax5.grid(True, alpha=0.3)
        
        # Hide the unused subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generate_stability_report(self, stability_metrics: LinearStabilityMetrics, 
                                save_plots: bool = True, plot_dir: str = "./plots/"):
        """Generate comprehensive stability analysis report with plots."""
        import os
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
        
        # Generate all plots
        fig1 = self.plot_eigenvalues(stability_metrics)
        fig2 = self.plot_mode_characteristics(stability_metrics)
        fig3 = self.plot_time_response(stability_metrics)
        
        if save_plots:
            fig1.savefig(f"{plot_dir}/eigenvalues.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{plot_dir}/mode_characteristics.png", dpi=300, bbox_inches='tight')
            fig3.savefig(f"{plot_dir}/time_responses.png", dpi=300, bbox_inches='tight')
        
        # Print numerical summary
        print("=" * 60)
        print("LINEAR STABILITY ANALYSIS REPORT")
        print("=" * 60)
        
        print("\nLONGITUDINAL MODES:")
        print(f"Short Period: ωn = {stability_metrics.nat_freq_short_period.value} rad/s, "
              f"ζ = {stability_metrics.damping_ratio_short_period.value}")
        print(f"Phugoid: ωn = {stability_metrics.nat_freq_phugoid.value} rad/s, "
              f"ζ = {stability_metrics.damping_ratio_phugoid.value}")
        
        print("\nLATERAL-DIRECTIONAL MODES:")
        print(f"Spiral: ωn = {stability_metrics.nat_freq_spiral.value} rad/s, "
              f"ζ = {stability_metrics.damping_ratio_spiral.value}")
        print(f"Dutch Roll: ωn = {stability_metrics.nat_freq_dutch_roll.value} rad/s, "
              f"ζ = {stability_metrics.damping_ratio_dutch_roll.value}")
        print(f"Roll Subsidence: ωn = {stability_metrics.nat_freq_roll.value} rad/s, "
                f"ζ = {stability_metrics.damping_ratio_roll.value}")
        
        
        plt.show()
        
        return fig1, fig2, fig3
    
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

    def _identify_lateral_modes(self, eig_vecs_real, eig_real, eig_imag):
        """
        Identify spiral, dutch roll, and roll subsidence modes based on eigenvector characteristics.
        
        Spiral: dominated by v (sideslip) and phi (roll angle) - indices 0, 3, slow/aperiodic
        Dutch Roll: dominated by p (roll rate) and r (yaw rate) - indices 1, 2 (oscillatory)
        Roll Subsidence: dominated by p (roll rate) - index 1 (non-oscillatory, fast)
        """
        # Initialize score arrays for each mode
        dutch_roll_scores = csdl.Variable(value=0, shape=(4,), name="dutch_roll_scores")
        spiral_scores = csdl.Variable(value=0, shape=(4,), name="spiral_scores")
        roll_scores = csdl.Variable(value=0, shape=(4,), name="roll_scores")
        
        for i in range(4):
            # Extract components
            v_component = csdl.absolute(eig_vecs_real[0, i])   # v component (index 0)
            p_component = csdl.absolute(eig_vecs_real[1, i])   # p component (index 1)
            r_component = csdl.absolute(eig_vecs_real[2, i])   # r component (index 2)
            phi_component = csdl.absolute(eig_vecs_real[3, i]) # phi component (index 3)
            
            # Calculate frequency and oscillatory characteristics
            freq = ((eig_real[i] ** 2 + eig_imag[i] ** 2) + 1e-10) ** 0.5
            oscillatory_factor = csdl.absolute(eig_imag[i]) / (freq + 1e-10)
            real_part = csdl.absolute(eig_real[i])
            
            # Dutch roll score: dominated by p and r, oscillatory, moderate frequency
            dutch_roll_score = (p_component + r_component) * (1 + 3 * oscillatory_factor) * freq
            dutch_roll_scores = dutch_roll_scores.set(csdl.slice[i], dutch_roll_score)
            
            # Spiral score: dominated by v and phi, very low frequency, non-oscillatory
            # Emphasize low frequency and non-oscillatory nature
            spiral_score = (v_component + phi_component) * (1 - oscillatory_factor) / (freq + 1e-10) * 10
            spiral_scores = spiral_scores.set(csdl.slice[i], spiral_score)
            
            # Roll subsidence score: dominated by p only, high frequency, non-oscillatory
            # Emphasize high p component, high frequency, and non-oscillatory
            roll_score = p_component * freq * (1 - oscillatory_factor) * (1 + 2 * real_part) / (v_component + phi_component + 1e-10)
            roll_scores = roll_scores.set(csdl.slice[i], roll_score)
        
        # First identify Dutch roll (most oscillatory with p+r dominance)
        dutch_roll_idx = np.argmax(dutch_roll_scores.value)
        
        # Remove dutch roll from other considerations
        spiral_scores_adj = spiral_scores.set(csdl.slice[dutch_roll_idx], 0.0)
        roll_scores_adj = roll_scores.set(csdl.slice[dutch_roll_idx], 0.0)
        
        # Identify roll subsidence (highest frequency, p-dominated, non-oscillatory)
        roll_idx_final = np.argmax(roll_scores_adj.value)
        
        # Remove roll from spiral consideration
        spiral_scores_adj = spiral_scores_adj.set(csdl.slice[roll_idx_final], 0.0)
        
        # Identify spiral (lowest frequency, v+phi dominated, non-oscillatory)
        spiral_idx_final = np.argmax(spiral_scores_adj.value)
        
        # Additional validation: ensure modes are properly distinguished
        # If roll and spiral have the same index, use frequency as tie-breaker
        if spiral_idx_final == roll_idx_final:
            # Find the remaining unassigned modes
            remaining_indices = [j for j in range(4) if j not in [dutch_roll_idx]]
            if len(remaining_indices) >= 2:
                # Sort remaining by frequency (roll should be faster than spiral)
                freq_values = []
                for j in remaining_indices:
                    freq_j = ((eig_real[j] ** 2 + eig_imag[j] ** 2) + 1e-10) ** 0.5
                    freq_values.append((freq_j.value, j))
                
                freq_values.sort(reverse=True)  # Highest frequency first
                roll_idx_final = freq_values[0][1]  # Highest frequency -> roll
                spiral_idx_final = freq_values[1][1]  # Lower frequency -> spiral
        
        return spiral_idx_final, dutch_roll_idx, roll_idx_final


    

class EigenValueOperation(csdl.CustomExplicitOperation):
    def __init__(self):
        super().__init__()

    def evaluate(self, mat):
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
