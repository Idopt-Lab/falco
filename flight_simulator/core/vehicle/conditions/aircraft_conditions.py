from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
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


    def __repr__(self):
        try:
            ac_states = self.quantities.ac_states
            u_val = ac_states.states.u.value
            v_val = ac_states.states.v.value
            w_val = ac_states.states.w.value
            p_val = ac_states.states.p.value
            q_val = ac_states.states.q.value
            r_val = ac_states.states.r.value
            alt_val = ac_states.axis.translation_from_origin.z.value
            density = ac_states.atmospheric_states.density.value
            return (f"{self.__class__.__name__} | u={u_val} m/s, v={v_val} m/s, "
                    f"w={w_val} m/s, p={p_val} rad/s, q={q_val} rad/s, r={r_val} rad/s, "
                    f"Altitude={alt_val} m, Density={density} kg/m^3>")
        except Exception:
            return f"<{self.__class__.__name__} representation not available>"
          

    def assemble_forces_moments(self):
        """Assemble forces and moments from the component."""

        total_forces, total_moments = self.component.compute_total_loads(
            fd_state=self.quantities.ac_states,
            controls=self.controls)

        return total_forces, total_moments

    


class CruiseCondition(Condition):
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
        self.quantities.ac_states = AircraftStates(axis=self.axis)
        atmos_states = self.quantities.ac_states.atmospheric_states
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
        self.quantities.ac_states = AircraftStates(axis=self.axis)
        atmos_states = self.quantities.ac_states.atmospheric_states
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
        self.axis = fd_axis.copy()
        self.component = component
        self.controls = controls
        self._setup_condition()

    def _setup_condition(self):
        hover_parameters = self.parameters
        u = v = w = p = q = r = phi = theta = psi = x = y = csdl.Variable(value=0.)
        z = hover_parameters.altitude
        self.axis.translation_from_origin.z = z
        self.quantities.ac_states = AircraftStates(axis=self.axis)
        atmos_states = self.quantities.ac_states.atmospheric_states
        self.axis.euler_angles.theta = theta
        self.axis.euler_angles.phi = phi
        self.axis.euler_angles.psi = psi
        self.axis.translation_from_origin.x = x
        self.axis.translation_from_origin.y = y
        self.quantities.ac_states = AircraftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis)


