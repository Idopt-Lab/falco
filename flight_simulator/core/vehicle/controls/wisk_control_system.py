from typing import List, Union
import numpy as np
import csdl_alpha as csdl
from dataclasses import dataclass
from flight_simulator.core.vehicle.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl

class WiskControl(VehicleControlSystem):

    @dataclass
    class ControlVector(csdl.VariableGroup):
        throttle_ob_left_fwd: csdl.Variable
        throttle_mid_left_fwd: csdl.Variable
        throttle_ib_left_fwd: csdl.Variable
        throttle_ob_right_fwd: csdl.Variable
        throttle_mid_right_fwd: csdl.Variable
        throttle_ib_right_fwd: csdl.Variable
        throttle_ob_left_aft: csdl.Variable
        throttle_mid_left_aft: csdl.Variable
        throttle_ib_left_aft: csdl.Variable
        throttle_ob_right_aft: csdl.Variable
        throttle_mid_right_aft: csdl.Variable
        throttle_ib_right_aft: csdl.Variable
        aileron_ob_left: csdl.Variable
        flap_mid_left: csdl.Variable
        flap_ib_left: csdl.Variable
        aileron_ob_right: csdl.Variable
        flap_mid_right: csdl.Variable
        flap_ib_right: csdl.Variable
        elevator: csdl.Variable
        rudder: csdl.Variable

    def __init__(self, symmetrical: bool = True):
        self.symmetrical = symmetrical

        self.elevator = ControlSurface('elevator', lb=-15, ub=15)
        self.rudder = ControlSurface('rudder', lb=-30, ub=30)
        if not symmetrical:
            self.aileron_ob_left = ControlSurface('aileron_ob_left', lb=-15, ub=15)
            self.aileron_ob_right = ControlSurface('aileron_ob_right', lb=-15, ub=15)
            self.flap_mid_left = ControlSurface('flap_mid_left', lb=-15, ub=0)
            self.flap_ib_left = ControlSurface('flap_ib_left', lb=-15, ub=0)
            self.flap_mid_right = ControlSurface('flap_mid_right', lb=-15, ub=0)
            self.flap_ib_right = ControlSurface('flap_ib_right', lb=-15, ub=0)
        else:
            self.aileron = ControlSurface('aileron', lb=-15, ub=15)
            self.flap = ControlSurface('flap', lb=-15, ub=0)
                
        self.motor_ob_left_fwd = PropulsiveControl('motor_ob_left_fwd')
        self.motor_mid_left_fwd = PropulsiveControl('motor_mid_left_fwd')
        self.motor_ib_left_fwd = PropulsiveControl('motor_ib_left_fwd')
        self.motor_ob_right_fwd = PropulsiveControl('motor_ob_right_fwd')
        self.motor_mid_right_fwd = PropulsiveControl('motor_mid_right_fwd')
        self.motor_ib_right_fwd = PropulsiveControl('motor_ib_right_fwd')
        self.motor_ob_left_aft = PropulsiveControl('motor_ob_left_aft')
        self.motor_mid_left_aft = PropulsiveControl('motor_mid_left_aft')
        self.motor_ib_left_aft = PropulsiveControl('motor_ib_left_aft')
        self.motor_ob_right_aft = PropulsiveControl('motor_ob_right_aft')
        self.motor_mid_right_aft = PropulsiveControl('motor_mid_right_aft')
        self.motor_ib_right_aft = PropulsiveControl('motor_ib_right_aft')

        if symmetrical:
            self.u = self.ControlVector(
                aileron_ob_left=self.aileron.deflection,
                flap_mid_left=self.flap.deflection,
                flap_ib_left=self.flap.deflection,
                aileron_ob_right=-self.aileron.deflection,
                flap_mid_right=self.flap.deflection,
                flap_ib_right=self.flap.deflection,
                throttle_ob_left_fwd=self.motor_ob_left_fwd.throttle,
                throttle_mid_left_fwd=self.motor_mid_left_fwd.throttle,
                throttle_ib_left_fwd=self.motor_ib_left_fwd.throttle,
                throttle_ob_right_fwd=self.motor_ob_right_fwd.throttle,
                throttle_mid_right_fwd=self.motor_mid_right_fwd.throttle,
                throttle_ib_right_fwd=self.motor_ib_right_fwd.throttle,
                throttle_ob_left_aft=self.motor_ob_left_aft.throttle,
                throttle_mid_left_aft=self.motor_mid_left_aft.throttle,
                throttle_ib_left_aft=self.motor_ib_left_aft.throttle,
                throttle_ob_right_aft=self.motor_ob_right_aft.throttle,
                throttle_mid_right_aft=self.motor_mid_right_aft.throttle,
                throttle_ib_right_aft=self.motor_ib_right_aft.throttle,
                elevator=self.elevator.deflection,
                rudder=self.rudder.deflection
            )
        else:
            self.u = self.ControlVector(
                aileron_ob_left=self.aileron_ob_left.deflection,
                flap_mid_left=self.flap_mid_left.deflection,
                flap_ib_left=self.flap_ib_left.deflection,
                aileron_ob_right=self.aileron_ob_right.deflection,
                flap_mid_right=self.flap_mid_right.deflection,
                flap_ib_right=self.flap_ib_right.deflection,
                throttle_ob_left_fwd=self.motor_ob_left_fwd.throttle,
                throttle_mid_left_fwd=self.motor_mid_left_fwd.throttle,
                throttle_ib_left_fwd=self.motor_ib_left_fwd.throttle,
                throttle_ob_right_fwd=self.motor_ob_right_fwd.throttle,
                throttle_mid_right_fwd=self.motor_mid_right_fwd.throttle,
                throttle_ib_right_fwd=self.motor_ib_right_fwd.throttle,
                throttle_ob_left_aft=self.motor_ob_left_aft.throttle,
                throttle_mid_left_aft=self.motor_mid_left_aft.throttle,
                throttle_ib_left_aft=self.motor_ib_left_aft.throttle,
                throttle_ob_right_aft=self.motor_ob_right_aft.throttle,
                throttle_mid_right_aft=self.motor_mid_right_aft.throttle,
                throttle_ib_right_aft=self.motor_ib_right_aft.throttle,
                elevator=self.elevator.deflection,
                rudder=self.rudder.deflection
            )
        if symmetrical:
            super().__init__(pitch_control=[self.elevator],
                             roll_control=[self.aileron],
                             yaw_control=[self.rudder],
                             throttle_control=[self.motor_ob_left_fwd, self.motor_mid_left_fwd, self.motor_ib_left_fwd,
                                               self.motor_ob_right_fwd, self.motor_mid_right_fwd, self.motor_ib_right_fwd,
                                               self.motor_ob_left_aft, self.motor_mid_left_aft, self.motor_ib_left_aft,
                                               self.motor_ob_right_aft, self.motor_mid_right_aft, self.motor_ib_right_aft])
        else:
            super().__init__(pitch_control=[self.elevator],
                             roll_control=[self.aileron_ob_left, self.aileron_ob_right],
                             yaw_control=[self.rudder],
                             throttle_control=[self.motor_ob_left_fwd, self.motor_mid_left_fwd, self.motor_ib_left_fwd,
                                               self.motor_ob_right_fwd, self.motor_mid_right_fwd, self.motor_ib_right_fwd,
                                               self.motor_ob_left_aft, self.motor_mid_left_aft, self.motor_ib_left_aft,
                                               self.motor_ob_right_aft, self.motor_mid_right_aft, self.motor_ib_right_aft])
            
    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']
    
    @property
    def lower_bounds(self):
        lb_elevator = self.elevator.lower_bound
        lb_rudder = self.rudder.lower_bound
        lb_throttle = self.motor_ob_left_fwd.lower_bound
        lb_flap = self.flap.lower_bound

        if self.symmetrical:
            lb_aileron_ob_left = self.aileron.lower_bound
            lb_aileron_ob_right = self.aileron.lower_bound
            return np.array([lb_elevator, lb_rudder, lb_throttle, lb_flap, lb_aileron_ob_left, lb_aileron_ob_right])
        else:
            lb_aileron_ob_left = self.aileron_ob_left.lower_bound
            lb_aileron_ob_right = self.aileron_ob_right.lower_bound
            lb_flap_mid_left = self.flap_mid_left.lower_bound
            lb_flap_ib_left = self.flap_ib_left.lower_bound
            lb_flap_mid_right = self.flap_mid_right.lower_bound
            lb_flap_ib_right = self.flap_ib_right.lower_bound
            lb_throttle_ob_left_fwd = self.motor_ob_left_fwd.lower_bound
            lb_throttle_mid_left_fwd = self.motor_mid_left_fwd.lower_bound
            lb_throttle_ib_left_fwd = self.motor_ib_left_fwd.lower_bound
            lb_throttle_ob_right_fwd = self.motor_ob_right_fwd.lower_bound
            lb_throttle_mid_right_fwd = self.motor_mid_right_fwd.lower_bound
            lb_throttle_ib_right_fwd = self.motor_ib_right_fwd.lower_bound
            lb_throttle_ob_left_aft = self.motor_ob_left_aft.lower_bound
            lb_throttle_mid_left_aft = self.motor_mid_left_aft.lower_bound
            lb_throttle_ib_left_aft = self.motor_ib_left_aft.lower_bound
            lb_throttle_ob_right_aft = self.motor_ob_right_aft.lower_bound
            lb_throttle_mid_right_aft = self.motor_mid_right_aft.lower_bound
            lb_throttle_ib_right_aft = self.motor_ib_right_aft.lower_bound
            return np.array([lb_elevator, lb_rudder, lb_throttle_ob_left_fwd, lb_throttle_mid_left_fwd, lb_throttle_ib_left_fwd,
                             lb_throttle_ob_right_fwd, lb_throttle_mid_right_fwd, lb_throttle_ib_right_fwd, lb_throttle_ob_left_aft,
                             lb_throttle_mid_left_aft, lb_throttle_ib_left_aft, lb_throttle_ob_right_aft, lb_throttle_mid_right_aft,
                             lb_throttle_ib_right_aft, lb_aileron_ob_left, lb_flap_mid_left, lb_flap_ib_left, lb_aileron_ob_right,
                             lb_flap_mid_right, lb_flap_ib_right])
        
    @property
    def upper_bounds(self):
        ub_elevator = self.elevator.upper_bound
        ub_rudder = self.rudder.upper_bound
        ub_throttle = self.motor_ob_left_fwd.upper_bound
        ub_flap = self.flap.upper_bound

        if self.symmetrical:
            ub_aileron_ob_left = self.aileron.upper_bound
            ub_aileron_ob_right = self.aileron.upper_bound
            return np.array([ub_elevator, ub_rudder, ub_throttle, ub_flap, ub_aileron_ob_left, ub_aileron_ob_right])
        else:
            ub_aileron_ob_left = self.aileron_ob_left.upper_bound
            ub_aileron_ob_right = self.aileron_ob_right.upper_bound
            ub_flap_mid_left = self.flap_mid_left.upper_bound
            ub_flap_ib_left = self.flap_ib_left.upper_bound
            ub_flap_mid_right = self.flap_mid_right.upper_bound
            ub_flap_ib_right = self.flap_ib_right.upper_bound
            ub_throttle_ob_left_fwd = self.motor_ob_left_fwd.upper_bound
            ub_throttle_mid_left_fwd = self.motor_mid_left_fwd.upper_bound
            ub_throttle_ib_left_fwd = self.motor_ib_left_fwd.upper_bound
            ub_throttle_ob_right_fwd = self.motor_ob_right_fwd.upper_bound
            ub_throttle_mid_right_fwd = self.motor_mid_right_fwd.upper_bound
            ub_throttle_ib_right_fwd = self.motor_ib_right_fwd.upper_bound
            ub_throttle_ob_left_aft = self.motor_ob_left_aft.upper_bound
            ub_throttle_mid_left_aft = self.motor_mid_left_aft.upper_bound
            ub_throttle_ib_left_aft = self.motor_ib_left_aft.upper_bound
            ub_throttle_ob_right_aft = self.motor_ob_right_aft.upper_bound
            ub_throttle_mid_right_aft = self.motor_mid_right_aft.upper_bound
            ub_throttle_ib_right_aft = self.motor_ib_right_aft.upper_bound
            return np.array([ub_elevator, ub_rudder, ub_throttle_ob_left_fwd, ub_throttle_mid_left_fwd, ub_throttle_ib_left_fwd,
                             ub_throttle_ob_right_fwd, ub_throttle_mid_right_fwd, ub_throttle_ib_right_fwd, ub_throttle_ob_left_aft,
                             ub_throttle_mid_left_aft, ub_throttle_ib_left_aft, ub_throttle_ob_right_aft, ub_throttle_mid_right_aft,
                             ub_throttle_ib_right_aft, ub_aileron_ob_left, ub_flap_mid_left, ub_flap_ib_left, ub_aileron_ob_right,
                             ub_flap_mid_right, ub_flap_ib_right])