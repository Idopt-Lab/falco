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
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem


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


class Condition:
    """General aircraft condition."""

    def __init__(self,
                 ac_states: AircraftStates,
                 controls: VehicleControlSystem) -> None:
        self.controls = controls
        self.ac_states = ac_states

    def __repr__(self):
        try:
            ac_states = self.ac_states
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
                    f"Altitude={alt_val} m, Density={density} kg/m^3")
        except Exception:
            return f"{self.__class__.__name__} representation not available"

    def assemble_forces_moments(self, component: Component):
        """Assemble forces and moments from the component."""

        total_forces, total_moments = component.compute_total_loads(
            fd_state=self.ac_states,
            controls=self.controls)

        return total_forces, total_moments


class CruiseCondition(Condition):
    """Cruise condition: intended for steady analyses.

    Note: cannot specify all parameters at once (e.g., cannot specify speed and mach_number simultaneously)
    """

    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 controls: VehicleControlSystem,
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
        super().__init__(ac_states, controls)

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

        mach_number = self.parameters.mach_number
        speed = self.parameters.speed
        time = self.parameters.time
        range = self.parameters.range

        self._atm = NRLMSIS2.Atmosphere()
        self._atmos_states = self._atm.evaluate(-self.parameters.altitude)

        if mach_number.value != 0 and range.value != 0:
            V = self._atmos_states.speed_of_sound * mach_number
            self.parameters.speed = V
            time = range / V
            self.parameters.time = time
        elif mach_number.value != 0 and time.value != 0:
            V = self._atmos_states.speed_of_sound * mach_number
            self.parameters.speed = V
            range = V * time
            self.parameters.range = range
        elif speed.value != 0 and range.value != 0:
            V = speed
            mach_number = V / self._atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            time = range / V
            self.parameters.time = time
        elif speed.value != 0 and time.value != 0:
            V = speed
            mach_number = V / self._atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            range = V * time
            self.parameters.range = range
        else:
            raise NotImplementedError
        u = V * csdl.cos(self.parameters.pitch_angle)
        w = V * csdl.sin(self.parameters.pitch_angle)

        ac_states = AircraftStates(axis=axis, u=u, v=v, w=w, p=p, q=q, r=r)
        return ac_states
