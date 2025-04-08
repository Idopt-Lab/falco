from flight_simulator.core.condition.condition import Condition
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.dynamics.trim_stability import LinearStabilityMetrics
from flight_simulator.core.vehicle.models.equations_of_motion.eom_model import SixDoFModel
from typing import Union
import csdl_alpha as csdl
from dataclasses import dataclass
import numpy as np
import copy
import warnings
import NRLMSIS2
from flight_simulator import ureg, Q_



@dataclass
class AircraftStateQuantities(csdl.VariableGroup):
    ac_states: AircraftStates = None
    ac_states_atmos: NRLMSIS2.Atmosphere = None
    ac_eom_model = None
    stability_analysis = None
    total_forces = None
    total_moments = None


@dataclass
class HoverParameters(csdl.VariableGroup):
    altitude: csdl.Variable
    time: csdl.Variable

    def define_checks(self):
        self.add_check('altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('time', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

    def _check_parameters(self, name, value):
        if self._metadata[name]['type'] is not None:
            if type(value) not in self._metadata[name]['type']:
                raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")
        if self._metadata[name]['variablize']:
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                value.add_tag(tag=str(value_si.units))
        if self._metadata[name]['shape'] is not None:
            if value.shape != self._metadata[name]['shape']:
                raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value


@dataclass
class ClimbParameters(csdl.VariableGroup):
    initial_altitude: csdl.Variable
    final_altitude: csdl.Variable
    pitch_angle: csdl.Variable
    climb_gradient: csdl.Variable
    rate_of_climb: csdl.Variable
    speed: csdl.Variable
    mach_number: csdl.Variable
    flight_path_angle: csdl.Variable
    time: csdl.Variable

    def define_checks(self):
        self.add_check('initial_altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('final_altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('pitch_angle', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('climb_gradient', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('rate_of_climb', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('speed', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('mach_number', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('flight_path_angle', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('time', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

    def _check_parameters(self, name, value):
        if self._metadata[name]['type'] is not None:
            if type(value) not in self._metadata[name]['type']:
                raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")
        if self._metadata[name]['variablize']:
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                value.add_tag(tag=str(value_si.units))
        if self._metadata[name]['shape'] is not None:
            if value.shape != self._metadata[name]['shape']:
                raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value


@dataclass
class CruiseParameters(csdl.VariableGroup):
    altitude: csdl.Variable
    speed: csdl.Variable
    mach_number: csdl.Variable
    pitch_angle: csdl.Variable
    range: csdl.Variable
    time: csdl.Variable

    def define_checks(self):
        self.add_check('altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('speed', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('mach_number', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('pitch_angle', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('range', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('time', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

    def _check_parameters(self, name, value):
        if self._metadata[name]['type'] is not None:
            if type(value) not in self._metadata[name]['type']:
                raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")
        if self._metadata[name]['variablize']:
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                value.add_tag(tag=str(value_si.units))
        if self._metadata[name]['shape'] is not None:
            if value.shape != self._metadata[name]['shape']:
                raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value


class AircraftCondition(Condition):
    """General aircraft condition."""
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 u: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 v: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 w: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 p: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad/s'),
                 q: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad/s'),
                 r: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad/s'),
                 component=None,
                 controls=None) -> None:
        self.axis = fd_axis.copy()
        self.controls = controls
        self.component = component
        self.quantities: AircraftStateQuantities = AircraftStateQuantities()
        self.quantities.ac_states = AircraftStates(
            axis=self.axis,
            u=u, v=v, w=w,
            p=p, q=q, r=r
        )
        self.quantities.ac_states_atmos = self.quantities.ac_states.atm.evaluate(
            self.quantities.ac_states.axis.translation_from_origin.z)


    def compute_eom_model(self, print_output: bool = False):
        eom_model = SixDoFModel()
        self.quantities.ac_eom_model = eom_model.evaluate(
            total_forces=self.component.quantities.total_forces,
            total_moments=self.component.quantities.total_moments,
            ac_states=self.quantities.ac_states,
            ac_mass_properties=self.component.quantities.mass_properties
        )
        if print_output:
            print('-----------------------------------')
            print('Aircraft Condition:', self.__class__.__name__, 'in the Flight Dynamics Axis:')
            print('Elevator Deflection:', self.controls.elevator.deflection.value, 'rad')
            print('Rudder Deflection:', self.controls.rudder.deflection.value, 'rad')
            print('dp_dt:', self.quantities.ac_eom_model.dp_dt.value, 'rad/s^2')
            print('dq_dt:', self.quantities.ac_eom_model.dq_dt.value, 'rad/s^2')
            print('dr_dt:', self.quantities.ac_eom_model.dr_dt.value, 'rad/s^2')
            print('du_dt:', self.quantities.ac_eom_model.du_dt.value, 'm/s^2')
            print('dv_dt:', self.quantities.ac_eom_model.dv_dt.value, 'm/s^2')
            print('dw_dt:', self.quantities.ac_eom_model.dw_dt.value, 'm/s^2')
            print('accel_norm:', self.quantities.ac_eom_model.accel_norm.value, 'm/s^2')
            print('-----------------------------------')
        return self.quantities.ac_eom_model

    def assemble_forces_moments(self, print_output: bool = False):
        """Assemble forces and moments from the component."""

        total_forces, total_moments = self.component.compute_total_loads(
            fd_state=self.quantities.ac_states,
            controls=self.controls)

        if print_output:
            print('-----------------------------------')
            print('Aircraft Condition:', self.__class__.__name__, 'in the Flight Dynamics Axis:')
            print('u:', self.quantities.ac_states.states.u.value, 'm/s')
            print('v:', self.quantities.ac_states.states.v.value, 'm/s')
            print('w:', self.quantities.ac_states.states.w.value, 'm/s')
            print('p:', self.quantities.ac_states.states.p.value, 'rad/s')
            print('q:', self.quantities.ac_states.states.q.value, 'rad/s')
            print('r:', self.quantities.ac_states.states.r.value, 'rad/s')
            print('phi:', self.quantities.ac_states.states.phi.value, 'rad')
            print('theta:', self.quantities.ac_states.states.theta.value, 'rad')
            print('psi:', self.quantities.ac_states.states.psi.value, 'rad')
            print('Altitude:', self.quantities.ac_states.axis.translation_from_origin.z.value, 'm')
            print('Atmospheric Density:', self.quantities.ac_states_atmos.density.value, 'kg/m^3')
            print("Total Forces:", total_forces.value, 'N')
            print("Total Moments:", total_moments.value, 'N*m')
            print('-----------------------------------')
        return total_forces, total_moments

    def perform_linear_stability_analysis(self, print_output: bool = False) -> LinearStabilityMetrics:
        
        m = self.component.quantities.mass_properties.mass.value
        inertia_tensor = self.component.quantities.mass_properties.inertia_tensor.inertia_tensor
        ixx = inertia_tensor.value[0, 0]
        iyy = inertia_tensor.value[1, 1]
        izz = inertia_tensor.value[2, 2]
        ixz = inertia_tensor.value[0, 2]
        ixy = inertia_tensor.value[0, 1]
        iyz = inertia_tensor.value[1, 2]
        
        total_forces = self.component.quantities.total_forces
        total_moments = self.component.quantities.total_moments
        if len(total_forces.shape) == 1:
            total_forces = csdl.reshape(self.component.quantities.total_forces, shape=(1, 3))
            total_moments = csdl.reshape(self.component.quantities.total_moments, shape=(1, 3))
        num_nodes = total_forces.shape[0]
        g0 = 9.81
        ac_states = self.quantities.ac_states
        for i in range(num_nodes):
            A_mat_L = csdl.Variable(shape=(4, 4), value=0.)

            u = ac_states.states.u
            v = ac_states.states.v
            w = ac_states.states.w
            p = ac_states.states.p
            q = ac_states.states.q
            r = ac_states.states.r
            theta = ac_states.states.theta

            # Aerodynamic Forces and Moments
            X = total_forces[i, 0]
            Y = total_forces[i, 1]
            Z = total_forces[i, 2]
            L = total_moments[i, 0]
            M = total_moments[i, 1]
            N = total_moments[i, 2]

            # Longitudinal Stability Derivatives
            
            X_u = csdl.derivative(ofs=X, wrts=u)
            X_w = csdl.derivative(ofs=X, wrts=w)
            X_q = csdl.derivative(ofs=X, wrts=q)
            Z_u = csdl.derivative(ofs=Z, wrts=u)
            Z_w = csdl.derivative(ofs=Z, wrts=w)
            Z_q = csdl.derivative(ofs=Z, wrts=q)
            M_u = csdl.derivative(ofs=M, wrts=u)
            M_w = csdl.derivative(ofs=M, wrts=w)
            M_q = csdl.derivative(ofs=M, wrts=q)

            # Longitudinal A Matrix
            A_mat_L = A_mat_L.set(csdl.slice[0, 0], X_u / m)
            A_mat_L = A_mat_L.set(csdl.slice[0, 1], X_w / m)
            A_mat_L = A_mat_L.set(csdl.slice[0, 3], -g0 * csdl.cos(theta))
            A_mat_L = A_mat_L.set(csdl.slice[1, 0], Z_u / m)
            A_mat_L = A_mat_L.set(csdl.slice[1, 1], Z_w / m)
            A_mat_L = A_mat_L.set(csdl.slice[1, 2], (Z_q + m * u) / m)
            A_mat_L = A_mat_L.set(csdl.slice[1, 3], -g0 * csdl.sin(theta))
            A_mat_L = A_mat_L.set(csdl.slice[2, 0], M_u / iyy)
            A_mat_L = A_mat_L.set(csdl.slice[2, 1], M_w / iyy)
            A_mat_L = A_mat_L.set(csdl.slice[2, 2], M_q / iyy)
            A_mat_L = A_mat_L.set(csdl.slice[3, 2], 1.)

            # Lateral-Directional Stability Derivatives
            Y_v = csdl.derivative(ofs=Y, wrts=v)
            Y_p = csdl.derivative(ofs=Y, wrts=p)
            Y_r = csdl.derivative(ofs=Y, wrts=r)
            L_v = csdl.derivative(ofs=L, wrts=v)
            L_p = csdl.derivative(ofs=L, wrts=p)
            L_r = csdl.derivative(ofs=L, wrts=r)
            N_v = csdl.derivative(ofs=N, wrts=v)
            N_p = csdl.derivative(ofs=N, wrts=p)
            N_r = csdl.derivative(ofs=N, wrts=r)

            # Lateral-Directional A Matrix
            xi = ixx*izz - ixz**2
            A_mat_LD = csdl.Variable(shape=(4, 4), value=0.)
            A_mat_LD = A_mat_LD.set(csdl.slice[0, 0], Y_v / m)
            A_mat_LD = A_mat_LD.set(csdl.slice[0, 1], Y_p / m)
            A_mat_LD = A_mat_LD.set(csdl.slice[0, 2], Y_r / m - u)
            A_mat_LD = A_mat_LD.set(csdl.slice[0, 3], g0 * csdl.cos(theta))
            A_mat_LD = A_mat_LD.set(csdl.slice[1, 0], 1/xi * (izz*L_v + ixz*N_v))
            A_mat_LD = A_mat_LD.set(csdl.slice[1, 1], 1/xi * (izz*L_p + ixz*N_p))
            A_mat_LD = A_mat_LD.set(csdl.slice[1, 2], 1/xi * (izz*L_r + ixz*N_r))
            A_mat_LD = A_mat_LD.set(csdl.slice[2, 0], 1/xi * (ixz*L_v + ixx*N_v))
            A_mat_LD = A_mat_LD.set(csdl.slice[2, 1], 1/xi * (ixz*L_p + ixx*N_p))
            A_mat_LD = A_mat_LD.set(csdl.slice[2, 2], 1/xi * (ixz*L_r + ixx*N_r))
            A_mat_LD = A_mat_LD.set(csdl.slice[3, 1], 1.)
            A_mat_LD = A_mat_LD.set(csdl.slice[3, 2], csdl.tan(theta))

                                
        eig_val_long_operation = EigenValueOperationLongitudinal()
        eig_real_long, eig_imag_long = eig_val_long_operation.evaluate(A_mat_L)
        eig_val_lat_operation = EigenValueOperationLateralDirectional()
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

        self.quantities.stability_analysis = LinearStabilityMetrics(
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
        if print_output:
            print('-----------------------------------')
            print('Longitudinal Stability Metrics for:', self.__class__.__name__, 'in the Flight Dynamics Axis:')
            print('A Matrix:', self.quantities.stability_analysis.A_mat_longitudinal.value)
            print('Short Period Real Eigenvalue:', self.quantities.stability_analysis.real_eig_short_period.value)
            print('Short Period Imaginary Eigenvalue:', self.quantities.stability_analysis.imag_eig_short_period.value)
            print('Short Period Natural Frequency:', self.quantities.stability_analysis.nat_freq_short_period.value)
            print('Short Period Damping Ratio:', self.quantities.stability_analysis.damping_ratio_short_period.value)
            print('Short Period Time to Double:', self.quantities.stability_analysis.time_2_double_short_period.value)
            print('Phugoid Real Eigenvalue:', self.quantities.stability_analysis.real_eig_phugoid.value)
            print('Phugoid Imaginary Eigenvalue:', self.quantities.stability_analysis.imag_eig_phugoid.value)
            print('Phugoid Natural Frequency:', self.quantities.stability_analysis.nat_freq_phugoid.value)
            print('Phugoid Damping Ratio:', self.quantities.stability_analysis.damping_ratio_phugoid.value)
            print('Phugoid Time to Double:', self.quantities.stability_analysis.time_2_double_phugoid.value)
            print('-----------------------------------')
        return self.quantities.stability_analysis


class EigenValueOperationLongitudinal(csdl.CustomExplicitOperation):
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


class EigenValueOperationLateralDirectional(csdl.CustomExplicitOperation):
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
class CruiseCondition(AircraftCondition):
    """Cruise condition: intended for steady analyses.

    Note: cannot specify all parameters at once (e.g., cannot specify speed and mach_number simultaneously)
    """
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 range: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 pitch_angle: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
                 speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 mach_number: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'dimensionless'),
                 time: Union[ureg.Quantity, csdl.Variable] = Q_(0, 's'),
                 component=None,
                 controls=None):
        self.parameters: CruiseParameters = CruiseParameters(
            altitude=altitude,
            speed=speed,
            range=range,
            pitch_angle=pitch_angle,
            mach_number=mach_number,
            time=time,
        )
        self.quantities = AircraftStateQuantities()
        self._atmos_model = NRLMSIS2.Atmosphere()
        self.axis = fd_axis.copy()
        self.component = component
        self.controls = controls
        self._setup_condition()

    def _setup_condition(self):
        conflicting_attributes_1 = ["speed", "mach_number"]
        conflicting_attributes_2 = ["speed", "time", "range"]
        conflicting_attributes_3 = ["mach_number", "time", "range"]
        if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_1):
            raise Exception("Cannot specify 'mach_number' and 'speed' at the same time")
        if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_2):
            raise Exception("Cannot specify 'speed', 'time', and 'range' at the same time")
        if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_3):
            raise Exception("Cannot specify 'mach_number', 'time', and 'range' at the same time")
        x = y = v = phi = psi = p = q = r = csdl.Variable(value=0.)
        z = self.parameters.altitude
        self.axis.translation_from_origin.z = z
        atmos_states = self._atmos_model.evaluate(z)
        theta = self.parameters.pitch_angle
        mach_number = self.parameters.mach_number
        speed = self.parameters.speed * self.controls.throttle
        time = self.parameters.time
        range = self.parameters.range
        if mach_number.value != 0 and range.value != 0:
            V = atmos_states.speed_of_sound * mach_number
            self.parameters.speed = V
            time = range / V
            self.parameters.time = time
        elif mach_number.value != 0 and time.value != 0:
            V = atmos_states.speed_of_sound * mach_number
            self.parameters.speed = V
            range = V * time
            self.parameters.range = range
        elif speed.value != 0 and range.value != 0:
            V = speed
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            time = range / V
            self.parameters.time = time
        elif speed.value != 0 and time.value != 0:
            V = speed
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            range = V * time
            self.parameters.range = range
        else:
            raise NotImplementedError
        u = V * csdl.cos(theta)
        w = V * csdl.sin(theta)
        self.axis.euler_angles.theta = theta
        self.axis.euler_angles.phi = phi
        self.axis.euler_angles.psi = psi
        self.axis.translation_from_origin.x = x
        self.axis.translation_from_origin.y = y
        self.quantities.ac_states = AircraftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis)
        self.quantities.ac_states_atmos = atmos_states
        if self.component is not None:
            self.quantities.total_forces, self.quantities.total_moments = self.assemble_forces_moments()
            self.quantities.ac_eom_model = self.compute_eom_model()
            self.quantities.stability_analysis = self.perform_linear_stability_analysis()



class ClimbCondition(AircraftCondition):
    """Climb condition (intended for steady analyses)"""
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 initial_altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 final_altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 pitch_angle: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
                 flight_path_angle: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
                 speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 mach_number: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'dimensionless'),
                 time: Union[ureg.Quantity, csdl.Variable] = Q_(0, 's'),
                 climb_gradient: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 rate_of_climb: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 component=None,
                 controls=None) -> None:
        self.parameters: ClimbParameters = ClimbParameters(
            initial_altitude=initial_altitude,
            final_altitude=final_altitude,
            pitch_angle=pitch_angle,
            speed=speed,
            mach_number=mach_number,
            flight_path_angle=flight_path_angle,
            time=time,
            rate_of_climb=rate_of_climb,
            climb_gradient=climb_gradient,
        )
        self.quantities = AircraftStateQuantities()
        self._atmos_model = NRLMSIS2.Atmosphere()
        self.axis = fd_axis.copy()
        self.component = component
        self.controls = controls
        self._setup_condition()

    def _setup_condition(self):
        conflicting_attributes_1 = ["speed", "mach_number"]
        conflicting_attributes_2 = ["speed", "time"]
        conflicting_attributes_3 = ["mach_number", "time"]
        if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_1):
            raise Exception("Cannot specify 'mach_number' and 'speed' at the same time")
        if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_2):
            raise Exception("Cannot specify 'speed' and 'time' at the same time")
        if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_3):
            raise Exception("Cannot specify 'mach_number' and 'time' at the same time")
        x = y = v = p = q = r = phi = psi = csdl.Variable(value=0.)
        hi = self.parameters.initial_altitude
        hf = self.parameters.final_altitude
        h_mean = 0.5 * (hi + hf)
        theta = self.parameters.pitch_angle
        gamma = self.parameters.flight_path_angle
        self.axis.translation_from_origin.z = h_mean
        atmos_states = self._atmos_model.evaluate(altitude=h_mean)
        mach_number = self.parameters.mach_number
        speed = self.parameters.speed
        time = self.parameters.time
        if mach_number.value != 0:
            V = mach_number * atmos_states.speed_of_sound
            self.parameters.speed = V
        elif speed.value != 0:
            V = speed
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
        else:
            w_val = (hf - hi) / time
            u_val = w_val / csdl.tan(gamma)
            V = (u_val**2 + w_val**2)**0.5
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            self.parameters.speed = V
        if time.value != 0:
            h = abs(hf - hi)
            d = h / csdl.tan(gamma)
            time = ((d**2 + h**2)**0.5) / V
            self.parameters.time = time
        alfa = theta - gamma
        u = V * csdl.cos(alfa)
        w = V * csdl.sin(alfa)
        self.axis.euler_angles.theta = theta
        self.axis.euler_angles.phi = phi
        self.axis.euler_angles.psi = psi
        self.axis.translation_from_origin.x = x
        self.axis.translation_from_origin.y = y
        self.quantities.ac_states = AircraftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis)
        self.quantities.ac_states_atmos = atmos_states
        if self.component is not None:
            self.quantities.total_forces, self.quantities.total_moments = self.assemble_forces_moments()
            self.quantities.ac_eom_model = self.compute_eom_model()
            self.quantities.stability_analysis = self.perform_linear_stability_analysis()




class HoverCondition(AircraftCondition):
    """Hover condition (intended for steady analyses)

    Parameters:
     - altitude: Value or csdl.Variable
     - time: Value or csdl.Variable
    """
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 time: Union[ureg.Quantity, csdl.Variable] = Q_(0, 's'),
                 component=None,
                 controls=None) -> None:
        self.parameters: HoverParameters = HoverParameters(
            altitude=altitude,
            time=time,
        )
        self.quantities = AircraftStateQuantities()
        self._atmos_model = NRLMSIS2.Atmosphere()
        self.axis = fd_axis.copy()
        self.component = component
        self.controls = controls
        self._setup_condition()

    def _setup_condition(self):
        hover_parameters = self.parameters
        u = v = w = p = q = r = phi = theta = psi = x = y = csdl.Variable(value=0.)
        z = hover_parameters.altitude
        self.axis.translation_from_origin.z = z
        atmos_states = self._atmos_model.evaluate(z)
        self.axis.euler_angles.theta = theta
        self.axis.euler_angles.phi = phi
        self.axis.euler_angles.psi = psi
        self.axis.translation_from_origin.x = x
        self.axis.translation_from_origin.y = y
        self.quantities.ac_states = AircraftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis)
        self.quantities.ac_states_atmos = atmos_states
        if self.component is not None:
            self.quantities.total_forces, self.quantities.total_moments = self.assemble_forces_moments()
            self.quantities.ac_eom_model = self.compute_eom_model()
            self.quantities.stability_analysis = self.perform_linear_stability_analysis()

