from dataclasses import dataclass
import numpy as np

from flight_simulator.core.vehicle.vehicle_control_system import (
    VehicleControlSystem, ControlSurface, PropulsiveControl)
from typing import List

import csdl_alpha as csdl

# Create a Mass Properties object with given values

# Create a component without geometry. Its just a point mass.

# Create Aerodynamic Model

# Create Propulsion Model



class C172Control(VehicleControlSystem):

    @dataclass
    class ControlVector(csdl.VariableGroup):
        aileron_left: csdl.Variable
        aileron_right: csdl.Variable
        elevator: csdl.Variable
        rudder: csdl.Variable
        throttle: csdl.Variable

    def __init__(self, symmetrical: bool = True):
        self.symmetrical = symmetrical

        self.elevator = ControlSurface('elevator_left', lb=-25, ub=25)
        if not symmetrical:
            self.aileron_left = ControlSurface('aileron_left', lb=-10, ub=10)
            self.aileron_right = ControlSurface('aileron_right', lb=-10, ub=10)
        else:
            self.aileron = ControlSurface('aileron', lb=-10, ub=10)
        self.rudder = ControlSurface('rudder', lb=-30, ub=30)

        self.engine = PropulsiveControl(name='engine')

        if symmetrical:
            self.u = self.ControlVector(
                aileron_left=self.aileron.deflection,
                aileron_right=-self.aileron.deflection,
                elevator=self.elevator.deflection,
                rudder=self.rudder.deflection,
                throttle=self.engine.throttle)
        else:
            self.u = self.ControlVector(
                aileron_left=self.aileron_left.deflection,
                aileron_right=self.aileron_right.deflection,
                elevator=self.elevator.deflection,
                rudder=self.rudder.deflection,
                throttle=self.engine.throttle)

        if symmetrical:
            super().__init__(pitch_control=[self.elevator],
                             roll_control=[self.aileron],
                             yaw_control=[self.rudder],
                             throttle_control=[self.engine])
        else:
            super().__init__(pitch_control=[self.elevator],
                             roll_control=[self.aileron_left, self.aileron_right],
                             yaw_control=[self.rudder],
                             throttle_control=[self.engine])

    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']

    @property
    def lower_bounds(self):
        lb_elevator = self.elevator.lower_bound
        lb_rudder = self.rudder.lower_bound
        lb_thr = self.engine.lower_bound

        if self.symmetrical:
            lb_aileron_left = self.aileron.lower_bound
            lb_aileron_right = self.aileron.lower_bound
            return np.array([lb_aileron_left, lb_aileron_right, lb_elevator, lb_rudder, lb_thr])
        else:
            lb_aileron_left = self.aileron_left.lower_bound
            lb_aileron_right = self.aileron_right.lower_bound
            return np.array([lb_aileron_left, lb_aileron_right, lb_elevator, lb_rudder, lb_thr])

    @property
    def upper_bounds(self):
        ub_elevator = self.elevator.upper_bound
        ub_rudder = self.rudder.upper_bound
        ub_thr = self.engine.upper_bound

        if self.symmetrical:
            ub_aileron_left = self.aileron.upper_bound
            ub_aileron_right = self.aileron.upper_bound
            return np.array([ub_aileron_left, ub_aileron_right, ub_elevator, ub_rudder, ub_thr])
        else:
            ub_aileron_left = self.aileron_left.upper_bound
            ub_aileron_right = self.aileron_right.upper_bound
            return np.array([ub_aileron_left, ub_aileron_right, ub_elevator, ub_rudder, ub_thr])