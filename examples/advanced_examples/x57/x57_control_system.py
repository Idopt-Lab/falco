from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl, PropulsiveControlRPM
import csdl_alpha as csdl
import numpy as np
from typing import List

class OnOffControlSurface:
    def __init__(self, name, lb, ub, component=None):
        self.name = name
        self._lb = lb
        self._ub = ub
        if component is None: # Massless control surface case with no deflection pre-defined
            self.deflection = csdl.Variable(name=name+'_deflection', shape=(1, ), value=0.)
        elif component is not None and component.parameters.actuate_angle is None:
            self.deflection = csdl.Variable(name=name+'_deflection', shape=(1, ), value=0.)
        else:
            assert isinstance(component.parameters.actuate_angle, csdl.Variable)
            self.deflection = component.parameters.actuate_angle
        self.flag = False

    @property
    def lower_bound(self):
        return self._lb

    @property
    def upper_bound(self):
        return self._ub

    def get_flag(self):
        return self._flag

    def set_flag(self, flag_bool: bool):
        assert isinstance(flag_bool, bool)
        self._flag = flag_bool

        

class Blower:
    def __init__(self, name='High lift blower', flag_bool=False):
        self.name = name
        self.flag = flag_bool

    def get_flag(self):
        return self._flag

    def set_flag(self, flag_bool: bool):
        assert isinstance(flag_bool, bool)
        self._flag = flag_bool


class X57ControlSystem(VehicleControlSystem):

    def __init__(self, elevator_component, trim_tab_component, aileron_right_component, aileron_left_component,
                 flap_left_component, flap_right_component,
                 rudder_component, hl_engine_count: int, cm_engine_count: int, high_lift_blower_component: Blower) -> None:

        self.elevator = ControlSurface('elevator', lb=-26, ub=28, component=elevator_component)
        self.trim_tab = ControlSurface('trim_tab', lb=-15, ub=20, component=trim_tab_component)
        self.aileron_left = ControlSurface('aileron_left', lb=-15, ub=20, component=aileron_left_component)
        self.aileron_right = ControlSurface('aileron_right', lb=-15, ub=20, component=aileron_right_component)
        self.flap_left = OnOffControlSurface('flap_left', lb=-15, ub=20, component=flap_left_component)
        self.flap_right = OnOffControlSurface('flap_right', lb=-15, ub=20, component=flap_right_component)
        self.rudder = ControlSurface('rudder', lb=-16, ub=16, component=rudder_component)

        self.hl_engines = self._init_hl_engines(hl_engine_count)
        midpoint_hl = len(self.hl_engines) // 2
        self.hl_engines_left = self.hl_engines[:midpoint_hl]
        self.hl_engines_right = self.hl_engines[midpoint_hl:]
        self.cm_engines = self._init_cm_engines(cm_engine_count)
        midpoint_cm = len(self.cm_engines) // 2
        self.cm_engines_left = self.cm_engines[:midpoint_cm]
        self.cm_engines_right = self.cm_engines[midpoint_cm:]
        self.engines = self.hl_engines + self.cm_engines



        self.pitch_control = {'Elevator': self.elevator, 'Trim Tab': self.trim_tab}
        self.roll_control = {'Left Aileron': self.aileron_left, 'Rigt Aileron': self.aileron_right}
        self.yaw_control = {'Rudder': self.rudder}
        self.throttle_control = {'Cruise Engines': self.cm_engines, 'High Lift Engines': self.hl_engines}
        self.high_lift_control = {'Left Flap': self.flap_left, 
                                  'Right Flap': self.flap_right, 
                                  'Blower': high_lift_blower_component}



    def _init_hl_engines(self, count: int) -> list:
        """Initialize high-lift engines."""
        #Found on pg. 7 of x57_DiTTo_manuscript-v6.pdf
        return [PropulsiveControl(name=f'HL_Motor{i + 1}', throttle=0.0) for i in range(count)]
    
    def _init_cm_engines(self, count: int) -> list:
        """Initialize cruise engines."""
        #Found on pg. 7 of x57_DiTTo_manuscript-v6.pdf
        return [PropulsiveControl(name=f'Cruise_Motor{i + 1}', throttle=0.0) for i in range(count)]

    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle_control']
    
    @property
    def u(self):
        self.throttle_control = tuple(engine.throttle for engine in self.engines)
        control = (
            self.roll_control['Left Aileron'].deflection,
            self.roll_control['Rigt Aileron'].deflection,
            self.pitch_control['Elevator'].deflection,
            self.pitch_control['Trim Tab'].deflection,
            self.yaw_control['Rudder'].deflection) +  self.throttle_control
        return control
    
    
    def update_controls(self, new_u_vec):
        self.aileron_left.deflection = new_u_vec[0]
        self.aileron_right.deflection = new_u_vec[1]
        self.elevator.deflection = new_u_vec[2]
        self.trim_tab.deflection = new_u_vec[3]
        self.rudder.deflection = new_u_vec[4]
        n_engines = len(self.engines)
        for i, engine in enumerate(self.engines):
            engine.throttle = new_u_vec[5 + i] if i < n_engines else 0.0

    

    @property
    def min_values(self):
        return np.array([
                            self.aileron_left.min_value,
                            self.aileron_right.min_value,
                            self.elevator.min_value,
                            self.trim_tab.min_value,
                            self.rudder.min_value
                        ] + [engine.min_value for engine in self.engines])

    @property
    def max_values(self):
        return np.array([
                            self.aileron_left.max_value,
                            self.aileron_right.max_value,
                            self.elevator.max_value,
                            self.trim_tab.max_value,
                            self.rudder.max_value
                        ] + [engine.max_value for engine in self.engines])
    
    def update_high_lift_control(self, flap_flag: bool, blower_flag: bool = None):
        # Instead of using a property setter for flag, call an explicit method.
        self.high_lift_control['Left Flap'].set_flag(flap_flag)
        self.high_lift_control['Right Flap'].set_flag(flap_flag)
        if 'Blower' in self.high_lift_control and blower_flag is not None:
            self.high_lift_control['Blower'].set_flag(blower_flag)
...