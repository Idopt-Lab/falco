from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
import csdl_alpha as csdl
import numpy as np
from typing import List


class X57ControlSystem(VehicleControlSystem):

    def __init__(self, elevator_component, trim_tab_component, aileron_right_component, aileron_left_component,
                 flap_left_component, flap_right_component,
                 rudder_component, hl_engine_count: int, cm_engine_count: int) -> None:

        self.elevator = ControlSurface('elevator', lb=-26, ub=28, component=elevator_component)
        self.trim_tab = ControlSurface('trim_tab', lb=-15, ub=20, component=trim_tab_component)
        self.aileron_left = ControlSurface('aileron_left', lb=-15, ub=20, component=aileron_left_component)
        self.aileron_right = ControlSurface('aileron_right', lb=-15, ub=20, component=aileron_right_component)
        self.flap_left = ControlSurface('flap_left', lb=-15, ub=20, component=flap_left_component)
        self.flap_right = ControlSurface('flap_right', lb=-15, ub=20, component=flap_right_component)
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

        control = (
            self.aileron_left.deflection,
            self.aileron_right.deflection,
            self.flap_left.deflection,
            self.flap_right.deflection,
            self.elevator.deflection,
            self.trim_tab.deflection,
            self.rudder.deflection
        )
        # Use all engine throttle values for control vector
        engine_controls = tuple(engine.throttle for engine in self.engines)
        self.u = csdl.concatenate(control + engine_controls, axis=0)


        super().__init__(
            pitch_control=[self.elevator, self.trim_tab],
            roll_control=[self.aileron_left, self.aileron_right],
            yaw_control=[self.rudder],
            throttle_control=self.engines
        )

    def _init_hl_engines(self, count: int) -> list:
        """Initialize high-lift engines."""
        return [PropulsiveControl(name=f'HL_Motor{i + 1}', throttle=1.0) for i in range(count)]

    def _init_cm_engines(self, count: int) -> list:
        """Initialize cruise engines."""
        return [PropulsiveControl(name=f'Cruise_Motor{i + 1}', throttle=1.0) for i in range(count)]

    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']

    @property
    def min_values(self):
        return np.array([
                            self.aileron_left.min_value,
                            self.aileron_right.min_value,
                            self.flap_left.min_value,
                            self.flap_right.min_value,
                            self.elevator.min_value,
                            self.rudder.min_value
                        ] + [engine.min_value for engine in self.engines])

    @property
    def max_values(self):
        return np.array([
                            self.aileron_left.max_value,
                            self.aileron_right.max_value,
                            self.flap_left.max_value,
                            self.flap_right.max_value,
                            self.elevator.max_value,
                            self.rudder.max_value
                        ] + [engine.max_value for engine in self.engines])
