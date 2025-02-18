from flight_simulator.core.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
from typing import List, Union

class AircraftControlSystem(VehicleControlSystem):

    @dataclass
    class ControlVector(csdl.VariableGroup):
        left_aileron: csdl.Variable
        right_aileron: csdl.Variable
        left_flap: csdl.Variable
        right_flap: csdl.Variable
        elevator: csdl.Variable
        rudder: csdl.Variable
        throttle: csdl.Variable

    def __init__(self, symmetrical: bool = True):
        self.symmetrical = symmetrical
        self.elevator = ControlSurface(name='Elevator', min_value=-25, max_value=25)
        
        if not symmetrical:
            self.left_aileron = ControlSurface(name='Left Aileron', min_value=-25, max_value=25)
            self.right_aileron = ControlSurface(name='Right Aileron', min_value=-25, max_value=25)
            self.left_flap = ControlSurface(name='Left Flap', min_value=-25, max_value=25)
            self.right_flap = ControlSurface(name='Right Flap', min_value=-25, max_value=25)
        else:
            self.aileron = ControlSurface(name='Aileron', min_value=-10, max_value=10)
            self.flap = ControlSurface(name='Flap', min_value=-25, max_value=25)

        self.rudder = ControlSurface(name='Rudder', min_value=-30, max_value=30)

        self.engine1 = PropulsiveControl(name='Motor1')
        self.engine2 = PropulsiveControl(name='Motor2')
        self.engine3 = PropulsiveControl(name='Motor3')
        self.engine4 = PropulsiveControl(name='Motor4')
        self.engine5 = PropulsiveControl(name='Motor5')
        self.engine6 = PropulsiveControl(name='Motor6')
        self.engine7 = PropulsiveControl(name='Motor7')
        self.engine8 = PropulsiveControl(name='Motor8')
        self.engine9 = PropulsiveControl(name='Motor9')
        self.engine10 = PropulsiveControl(name='Motor10')
        self.engine11 = PropulsiveControl(name='Motor11')
        self.engine12 = PropulsiveControl(name='Motor12')

        if symmetrical:
            self.u = self.ControlVector(
                left_aileron = self.aileron.deflection,
                right_aileron = -self.aileron.deflection,
                left_flap = self.flap.deflection,
                right_flap = -self.flap.deflection,
                elevator = self.elevator.deflection,
                rudder = self.rudder.deflection,
                throttle = self.engine1.throttle
                )
        else:
            self.u = self.ControlVector(
                left_aileron = self.left_aileron.deflection,
                right_aileron = self.right_aileron.deflection,
                left_flap = self.left_flap.deflection,
                right_flap = self.right_flap.deflection,
                elevator = self.elevator.deflection,
                rudder = self.rudder.deflection,
                throttle1 = self.engine1.throttle,
                throttle2 = self.engine2.throttle,
                throttle3 = self.engine3.throttle,
                throttle4 = self.engine4.throttle,
                throttle5 = self.engine5.throttle,
                throttle6 = self.engine6.throttle,
                throttle7 = self.engine7.throttle,
                throttle8 = self.engine8.throttle,
                throttle9 = self.engine9.throttle,
                throttle10 = self.engine10.throttle,
                throttle11 = self.engine11.throttle,
                throttle12 = self.engine12.throttle
                )
        
        if symmetrical:
            super().__init__(
                pitch=[self.elevator],
                roll=[self.aileron],
                yaw=[self.rudder],
                throttle=[self.engine1]
            )
        else:
            super().__init__(
                pitch=[self.elevator],
                roll=[self.left_aileron, self.right_aileron],
                yaw=[self.rudder],
                throttle=[self.engine1, self.engine2, self.engine3, self.engine4, self.engine5, self.engine6, self.engine7, self.engine8, self.engine9, self.engine10, self.engine11, self.engine12]
            )
    @property
    def control_order(self)-> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']
    
    @property
    def min_values(self):
        min_elevator = self.elevator.min_value
        min_rudder = self.rudder.min_value
        min_aileron = self.aileron.min_value
        min_flap = self.flap.min_value
        min_throttle = self.engine1.min_value

        if self.symmetrical:
            min_left_aileron = self.aileron.min_value
            min_right_aileron = self.aileron.min_value
            return np.array([min_left_aileron, min_right_aileron, min_elevator, min_rudder, min_throttle])
        else:
            min_left_aileron = self.left_aileron.min_value
            min_right_aileron = self.right_aileron.min_value
            min_left_flap = self.left_flap.min_value
            min_right_flap = self.right_flap.min_value
            min_throttle1 = self.engine1.min_value
            min_throttle2 = self.engine2.min_value
            min_throttle3 = self.engine3.min_value
            min_throttle4 = self.engine4.min_value
            min_throttle5 = self.engine5.min_value
            min_throttle6 = self.engine6.min_value
            min_throttle7 = self.engine7.min_value
            min_throttle8 = self.engine8.min_value
            min_throttle9 = self.engine9.min_value
            min_throttle10 = self.engine10.min_value
            min_throttle11 = self.engine11.min_value
            min_throttle12 = self.engine12.min_value
            return np.array([min_left_aileron, min_right_aileron, min_left_flap, min_right_flap, min_elevator, min_rudder, min_throttle1, min_throttle2, min_throttle3, min_throttle4, min_throttle5, min_throttle6, min_throttle7, min_throttle8, min_throttle9, min_throttle10, min_throttle11, min_throttle12])
    @property
    def max_values(self):
        max_elevator = self.elevator.max_value
        max_rudder = self.rudder.max_value
        max_aileron = self.aileron.max_value
        max_flap = self.flap.max_value
        max_throttle = self.engine1.max_value

        if self.symmetrical:
            max_left_aileron = self.aileron.max_value
            max_right_aileron = self.aileron.max_value
            return np.array([max_left_aileron, max_right_aileron, max_elevator, max_rudder, max_throttle])
        else:
            max_left_aileron = self.left_aileron.max_value
            max_right_aileron = self.right_aileron.max_value
            max_left_flap = self.left_flap.max_value
            max_right_flap = self.right_flap.max_value
            max_throttle1 = self.engine1.max_value
            max_throttle2 = self.engine2.max_value
            max_throttle3 = self.engine3.max_value
            max_throttle4 = self.engine4.max_value
            max_throttle5 = self.engine5.max_value
            max_throttle6 = self.engine6.max_value
            max_throttle7 = self.engine7.max_value
            max_throttle8 = self.engine8.max_value
            max_throttle9 = self.engine9.max_value
            max_throttle10 = self.engine10.max_value
            max_throttle11 = self.engine11.max_value
            max_throttle12 = self.engine12.max_value
            return np.array([max_left_aileron, max_right_aileron, max_left_flap, max_right_flap, max_elevator, max_rudder, max_throttle1, max_throttle2, max_throttle3, max_throttle4, max_throttle5, max_throttle6, max_throttle7, max_throttle8, max_throttle9, max_throttle10, max_throttle11, max_throttle12])