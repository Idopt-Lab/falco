from flight_simulator.core.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
from flight_simulator import ureg, Q_
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

    def __init__(self, engine_count, symmetrical: bool = True):
        self.symmetrical = symmetrical
        self.elevator = ControlSurface(name='Elevator',lb=-26, ub=28)
        
        if not symmetrical:
            self.left_aileron = ControlSurface(name='Left Aileron',lb=-15, ub=20)
            self.right_aileron = ControlSurface(name='Right Aileron',lb=-15, ub=20)
            self.left_flap = ControlSurface(name='Left Flap',lb=-15, ub=20)
            self.right_flap = ControlSurface(name='Right Flap',lb=-15, ub=20)
        else:
            self.aileron = ControlSurface(name='Aileron',lb=-15, ub=20)
            self.flap = ControlSurface(name='Flap',lb=-15, ub=20)

        self.rudder = ControlSurface(name='Rudder',lb=-15, ub=15)
        num_engines = engine_count
        self.engines = [PropulsiveControl(name=f'Motor{i+1}', throttle=1.0) for i in range(num_engines)]

        if symmetrical:
            self.u = csdl.concatenate((self.aileron.deflection,
                                        -self.aileron.deflection,
                                        self.flap.deflection,
                                        -self.flap.deflection,
                                        self.elevator.deflection,
                                        self.rudder.deflection,
                                        self.engines[0].throttle),axis=0)
        else:
            self.u = csdl.concatenate((self.left_aileron.deflection,
                                       self.right_aileron.deflection,
                                       self.left_flap.deflection,
                                       self.right_flap.deflection,
                                       self.elevator.deflection,
                                       self.rudder.deflection,
                                       [engine.throttle for engine in self.engines]),axis=0)
        
        if symmetrical:
            super().__init__(
                pitch_control=[self.elevator],
                roll_control=[self.aileron],
                yaw_control=[self.rudder],
                throttle_control=[self.engines[0]]
            )
        else:
            super().__init__(
                pitch_control=[self.elevator],
                roll_control=[self.left_aileron, self.right_aileron],
                yaw_control=[self.rudder],
                throttle_control=[self.engines]
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
        min_throttle = self.engines[0].min_value

        if self.symmetrical:
            min_left_aileron = self.aileron.min_value
            min_right_aileron = self.aileron.min_value
            return np.array([min_left_aileron, min_right_aileron, min_elevator, min_rudder]+min_throttle)
        else:
            min_left_aileron = self.left_aileron.min_value
            min_right_aileron = self.right_aileron.min_value
            min_left_flap = self.left_flap.min_value
            min_right_flap = self.right_flap.min_value
            min_throttle = [engine.min_value for engine in self.engines]
            return np.array([min_left_aileron, min_right_aileron, min_left_flap, min_right_flap, min_elevator, min_rudder]+min_throttle)
    @property
    def max_values(self):
        max_elevator = self.elevator.max_value
        max_rudder = self.rudder.max_value
        max_aileron = self.aileron.max_value
        max_flap = self.flap.max_value
        max_throttle = self.engines[0].max_value

        if self.symmetrical:
            max_left_aileron = self.aileron.max_value
            max_right_aileron = self.aileron.max_value
            return np.array([max_left_aileron, max_right_aileron, max_elevator, max_rudder]+max_throttle)
        else:
            max_left_aileron = self.left_aileron.max_value
            max_right_aileron = self.right_aileron.max_value
            max_left_flap = self.left_flap.max_value
            max_right_flap = self.right_flap.max_value
            max_throttle = [engine.max_value for engine in self.engines]
            return np.array([max_left_aileron, max_right_aileron, max_left_flap, max_right_flap, max_elevator, max_rudder]+max_throttle)

