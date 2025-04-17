
from abc import abstractmethod
import numpy as np
import csdl_alpha as csdl
import numpy as np
from typing import Union
from dataclasses import dataclass
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem
from flight_simulator.core.vehicle.models.equations_of_motion.EoM import EquationsOfMotion
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.mass_properties import MassMI



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

class LinearStabilityAnalysis():

    def linear_stab_analysis(self, A, B):
        """
        Perform longitudinal linear stability analysis on a specifed aircraft state
        """

        # Ensure A and B are defined
        if A is None or B is None:
            raise ValueError("Matrices A and B must be defined for linear stability analysis.")
        
        # Long: u, w, q, theta
        A_mat_L = csdl.Variable(shape=(4, 4), name="A_mat_Longitudinal")
        long_indices=[0, 2, 4, 7]
        for i in csdl.frange(len(long_indices)):
            A_mat_L = A_mat_L.set(csdl.slice[i, :], A[long_indices[i], :])
            
        # Lat-Dir: v, p, r, 
        A_mat_LD = csdl.Variable(shape=(4, 4), name="A_mat_Lateral_Directional")
        latdir_indices=[1, 3, 5, 6]
        for i in csdl.frange(len(latdir_indices)):
            A_mat_LD = A_mat_LD.set(csdl.slice[i, :], A[latdir_indices[i], :])
        
            
        eig_val_long_operation = EigenValueOperation()
        eig_real_long, eig_imag_long = eig_val_long_operation.evaluate(A_mat_L)
        eig_val_lat_operation = EigenValueOperation()
        eig_real_lat, eig_imag_lat = eig_val_lat_operation.evaluate(A_mat_LD)
        
        # Short period
        lambda_sp_real = eig_real_long[0]
        lambda_sp_imag = eig_imag_long[0]
        sp_omega_n = ((lambda_sp_real ** 2 + lambda_sp_imag ** 2) + 1e-10) ** 0.5
        sp_damping_ratio = -lambda_sp_real / sp_omega_n
        sp_time_2_double = np.log(2) / ((lambda_sp_real ** 2 + 1e-10) ** 0.5)

        # Phugoid
        lambda_phugoid_real = eig_real_long[1]
        lambda_phugoid_imag = eig_imag_long[1]
        phugoid_omega_n = ((lambda_phugoid_real ** 2 + lambda_phugoid_imag ** 2) + 1e-10) ** 0.5
        phugoid_damping_ratio = -lambda_phugoid_real / phugoid_omega_n
        phugoid_time_2_double = np.log(2) / ((lambda_phugoid_real ** 2 + 1e-10) ** 0.5)

        # Spiral
        lambda_spiral_real = eig_real_lat[0]
        lambda_spiral_imag = eig_imag_lat[0]
        spiral_omega_n = ((lambda_spiral_real ** 2 + lambda_spiral_imag ** 2) + 1e-10) ** 0.5
        spiral_damping_ratio = -lambda_spiral_real / spiral_omega_n
        spiral_time_2_double = np.log(2) / ((lambda_spiral_real ** 2 + 1e-10) ** 0.5)

        # Dutch Roll
        lambda_dutch_roll_real = eig_real_lat[1]
        lambda_dutch_roll_imag = eig_imag_lat[1]
        dutch_roll_omega_n = ((lambda_dutch_roll_real ** 2 + lambda_dutch_roll_imag ** 2) + 1e-10) ** 0.5
        dutch_roll_damping_ratio = -lambda_dutch_roll_real / dutch_roll_omega_n
        dutch_roll_time_2_double = np.log(2) / ((lambda_dutch_roll_real ** 2 + 1e-10) ** 0.5)


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
            time_2_double_dutch_roll=dutch_roll_time_2_double
        )
        
        return stability_analysis
    

class EigenValueOperation(csdl.CustomExplicitOperation):
    def __init__(self):
        super().__init__()

    def evaluate(self, mat):
        shape = mat.shape
        size = shape[0]

        self.declare_input("mat", mat)
        eig_real = self.create_output("eig_vals_real", shape=(size, ))
        eig_imag = self.create_output("eig_vals_imag", shape=(size, ))

        self.declare_derivative_parameters("eig_vals_real", "mat")
        self.declare_derivative_parameters("eig_vals_imag", "mat")

        return eig_real, eig_imag

    def compute(self, inputs, outputs):
        mat = inputs["mat"]

        eig_vals, eig_vecs = np.linalg.eig(mat)
    
        idx = np.abs(eig_vals).argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        outputs['eig_vals_real'] = np.real(eig_vals)
        outputs['eig_vals_imag'] = np.imag(eig_vals)

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
        temp_r = np.zeros((size, size*size))
        temp_i = np.zeros((size, size*size))

        for j in range(len(eig_vals)):
            # dA/dw(j,:) = v(:,j)*(v^-T)(:j)
            partial = np.outer(eig_vecs[:, j], v_inv_T[:, j]).flatten(order='F')
            # Note that the order of flattening matters, hence argument in flatten()

            # Set jacobian rows
            temp_r[j, :] = np.real(partial)
            temp_i[j, :] = np.imag(partial)

        # Set Jacobian
        derivatives['eig_vals_real', 'mat'] = temp_r
        derivatives['eig_vals_imag', 'mat'] = temp_i