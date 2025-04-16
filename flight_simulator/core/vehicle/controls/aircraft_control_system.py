from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
from typing import List
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl



class AircraftControlSystem(VehicleControlSystem):

    def __init__(self, engine_count: int, symmetrical: bool = True)-> None:
        self.symmetrical = symmetrical
        self.elevator = ControlSurface(name='Elevator',lb=-26, ub=28)
        
        if symmetrical:
            self._init_symmetrical_controls()
        else:
            self._init_asymmetrical_controls()

        self.rudder = ControlSurface(name='Rudder',lb=-15, ub=15)
        self.engines = self._init_engines(engine_count)
        self.u = self._assemble_control_vector()


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
                throttle_control=self.engines
            )


    def _init_symmetrical_controls(self) -> None:
        """Initialize controls for a symmetrical configuration."""
        self.aileron = ControlSurface(name='Aileron', lb=-15, ub=20)
        self.flap = ControlSurface(name='Flap', lb=-15, ub=20)
    
    def _init_asymmetrical_controls(self) -> None:
        """Initialize controls for an asymmetrical configuration."""
        self.left_aileron = ControlSurface(name='Left Aileron', lb=-15, ub=20)
        self.right_aileron = ControlSurface(name='Right Aileron', lb=-15, ub=20)
        self.left_flap = ControlSurface(name='Left Flap', lb=-15, ub=20)
        self.right_flap = ControlSurface(name='Right Flap', lb=-15, ub=20)
    

    def _init_engines(self, engine_count: int) -> List[PropulsiveControl]:
        """
        Initialize the propulsion controls.

        Returns:
            List[PropulsiveControl]: List of propulsive control instances.
        """
        if self.symmetrical:
            common_throttle = PropulsiveControl(name='Motor', throttle=1)
            return [common_throttle for _ in range(engine_count)]
        else:
            return [PropulsiveControl(name=f'Motor{i+1}', throttle=1.0) for i in range(engine_count)]
        


    def _assemble_control_vector(self):
        """
        Assemble the control vector using the deflections of the surfaces.
        
        Returns:
            The concatenated control vector.
        """
        if self.symmetrical:
            return csdl.concatenate((
                self.aileron.deflection,
                -self.aileron.deflection,
                self.flap.deflection,
                -self.flap.deflection,
                self.elevator.deflection,
                self.rudder.deflection,
                self.engines[0].throttle
            ), axis=0)
        else:
            return csdl.concatenate((
                self.left_aileron.deflection,
                self.right_aileron.deflection,
                self.left_flap.deflection,
                self.right_flap.deflection,
                self.elevator.deflection,
                self.rudder.deflection
            ) + tuple(engine.throttle for engine in self.engines), axis=0)
        
    
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

