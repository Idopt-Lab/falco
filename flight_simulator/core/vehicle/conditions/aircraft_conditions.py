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


class Condition():
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
        #TODO: below is redundant with self.quantities.ac_states.atmospheric_states
        # self.quantities.ac_states_atmos = self.quantities.ac_states.atm.evaluate(
        #     self.quantities.ac_states.axis.translation_from_origin.z)


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

    


<<<<<<< Updated upstream
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
class CruiseCondition(Condition):
=======
class CruiseCondition(AircraftCondition):
>>>>>>> Stashed changes
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
        #TODO:  REDUNDANT BELOW
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




class ClimbCondition(Condition):
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





class HoverCondition(Condition):
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


