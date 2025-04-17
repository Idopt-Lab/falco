from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.vehicle.controls.aircraft_control_system import AircraftControlSystem
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.vehicle.components.component import Component
from typing import Union
import csdl_alpha as csdl
from dataclasses import dataclass
import numpy as np
import copy
import warnings
import NRLMSIS2
from flight_simulator import ureg, Q_
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem
from flight_simulator.core.dynamics.linear_stability import LinearStabilityAnalysis
from flight_simulator.core.vehicle.models.equations_of_motion.EoM import EquationsOfMotion


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
                 states: AircraftStates,
                 controls: VehicleControlSystem,
                 eom: EquationsOfMotion,
                 analysis: LinearStabilityAnalysis) -> None:
        self.ac_states = states
        self.controls = controls
        self.eom = eom
        self.analysis = analysis

    def __repr__(self):
        try:
            ac_states = self.ac_states
            u_val = ac_states.state_vector.u.value
            v_val = ac_states.state_vector.v.value
            w_val = ac_states.state_vector.w.value
            p_val = ac_states.state_vector.p.value
            q_val = ac_states.state_vector.q.value
            r_val = ac_states.state_vector.r.value
            alt_val = ac_states.axis.translation_from_origin.z.value
            density = ac_states.atmospheric_states.density.value
            return (f"{self.__class__.__name__} | u={u_val} m/s, v={v_val} m/s, "
                    f"w={w_val} m/s, p={p_val} rad/s, q={q_val} rad/s, r={r_val} rad/s, "
                    f"Altitude={alt_val} m, Density={density} kg/m^3")
        except Exception:
            return f"{self.__class__.__name__} representation not available"
          

    def assemble_forces_moments(self, component: Component):
        """Assemble forces and moments from the component."""

        total_forces, total_moments = component.compute_total_loads(
            fd_state=self.ac_states,
            controls=self.controls)

        return total_forces, total_moments
    
    def evaluate_eom(self, component: Component):
        
        tf, tm = self.assemble_forces_moments(component=component)
        mp = component.quantities.mass_properties
        st = self.ac_states

        r, x = self.eom._EoM_res(aircraft_states=st, mass_properties=mp, total_forces=tf, total_moments=tm)
        return r, x
    
    def evaluate_trim_res(self, component: Component):

        res, x = self.evaluate_eom(component=component)
        res_vec = csdl.Variable(shape=(6,), value=0.)
        res_vec = res_vec.set(csdl.slice[0:5], csdl.get_index(res,slices=csdl.slice[0:5]))
        J = csdl.norm(res_vec)

        return J

    def generate_linear_system(self, component: Component):
        """Conducts a Linear Stability Analysis."""
        
        r, x = self.evaluate_eom(component=component)
        # state_vector_list = [self.ac_states.state_vector.u, self.ac_states.state_vector.v, self.ac_states.state_vector.w,
        #                      self.ac_states.state_vector.p, self.ac_states.state_vector.q, self.ac_states.state_vector.r,
        #                      self.ac_states.state_vector.phi, self.ac_states.state_vector.theta, self.ac_states.state_vector.psi,
        #                      self.ac_states.state_vector.x, self.ac_states.state_vector.y, self.ac_states.state_vector.z]
        A = csdl.derivative(ofs=r, wrts=x)
        B = csdl.derivative(ofs=r, wrts=self.controls.u)

        return A, B
    
    def eval_linear_stability(self, component: Component):
        """Evaluates the linear stability analysis."""
        
        A, B = self.generate_linear_system(component=component)
        stability_metrics = self.analysis.linear_stab_analysis(A=A, B=B)

        return stability_metrics
        
    

class CruiseCondition(Condition):
    """Cruise condition: intended for steady analyses.

    Note: cannot specify all parameters at once (e.g., cannot specify speed and mach_number simultaneously)
    """
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 controls: AircraftControlSystem,
                 altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 range: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 pitch_angle: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
                 speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 mach_number: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'dimensionless'),
                 time: Union[ureg.Quantity, csdl.Variable] = Q_(0, 's')):

        self.parameters: CruiseParameters = CruiseParameters(
            altitude=altitude,
            speed=speed,
            range=range,
            pitch_angle=pitch_angle,
            mach_number=mach_number,
            time=time,
        )

        ac_states = self._setup_condition(fd_axis)
        eom = EquationsOfMotion()
        analysis = LinearStabilityAnalysis()
        super().__init__(states=ac_states, controls=controls, eom=eom, analysis=analysis)

    def _setup_condition(self, fd_axis: Union[Axis, AxisLsdoGeo]):
        axis = fd_axis.copy(new_name="cruise_axis")

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
        
        axis.translation_from_origin.x = x
        axis.translation_from_origin.y = y
        axis.translation_from_origin.z = self.parameters.altitude

        axis.euler_angles.phi = phi
        axis.euler_angles.theta = self.parameters.pitch_angle
        axis.euler_angles.psi = psi

        ac_states = AircraftStates(axis=axis)
        atmos_states = ac_states.atmospheric_states
        mach_number = self.parameters.mach_number
        speed = self.parameters.speed
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
        u = V * csdl.cos(self.parameters.pitch_angle)
        w = V * csdl.sin(self.parameters.pitch_angle)
        ac_states = AircraftStates(axis=axis, u=u, v=v, w=w, p=p, q=q, r=r)
        return ac_states




class ClimbCondition(Condition):
    """Climb condition (intended for steady analyses)"""
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 controls: AircraftControlSystem,
                 initial_altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 final_altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 pitch_angle: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
                 flight_path_angle: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
                 speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 mach_number: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'dimensionless'),
                 time: Union[ureg.Quantity, csdl.Variable] = Q_(0, 's'),
                 climb_gradient: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 rate_of_climb: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s')):
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

        ac_states = self._setup_condition(fd_axis)
        super().__init__(ac_states=ac_states, controls=controls)

    def _setup_condition(self, fd_axis: Union[Axis, AxisLsdoGeo]):
        axis = fd_axis.copy(new_name="climb_axis")
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

        axis.translation_from_origin.x = x
        axis.translation_from_origin.y = y
        axis.translation_from_origin.z = h_mean

        axis.euler_angles.phi = phi
        axis.euler_angles.theta = self.parameters.pitch_angle
        axis.euler_angles.psi = psi

        ac_states = AircraftStates(axis=axis)
        atmos_states = ac_states.atmospheric_states
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
        ac_states = AircraftStates(axis=axis, u=u, v=v, w=w, p=p, q=q, r=r)
        return ac_states




class HoverCondition(Condition):
    """Hover condition (intended for steady analyses)

    Parameters:
     - altitude: Value or csdl.Variable
     - time: Value or csdl.Variable
    """
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 controls: AircraftControlSystem,
                 altitude: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 time: Union[ureg.Quantity, csdl.Variable] = Q_(0, 's')):
        self.parameters: HoverParameters = HoverParameters(
            altitude=altitude,
            time=time,
        )
        ac_states = self._setup_condition(fd_axis)
        super().__init__(ac_states=ac_states, controls=controls)

    def _setup_condition(self, fd_axis: Union[Axis, AxisLsdoGeo]):
        axis = fd_axis.copy(new_name="hover_axis")
        hover_parameters = self.parameters
        u = v = w = p = q = r = phi = theta = psi = x = y = csdl.Variable(value=0.)

        axis.translation_from_origin.x = x
        axis.translation_from_origin.y = y
        axis.translation_from_origin.z = hover_parameters.altitude

        axis.euler_angles.phi = phi
        axis.euler_angles.theta = theta
        axis.euler_angles.psi = psi

        ac_states = AircraftStates(axis=axis)
        atmos_states = ac_states.atmospheric_states
        ac_states = AircraftStates(axis=axis, u=u, v=v, w=w, p=p, q=q, r=r)
        return ac_states


