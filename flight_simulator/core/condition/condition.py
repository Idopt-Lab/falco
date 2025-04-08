from typing import Union

from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem

class Condition:
    """The Condition class"""
    def __init__(self, fd_axis: Union[Axis, AxisLsdoGeo], control_system: VehicleControlSystem,
                  component: Component) -> None:
        self.parameters : dict = {}
        self.fd_axis = fd_axis
        self.controls = control_system
        self._component = component

    @property
    def component(self):
        return self._component
    
    @component.setter
    def component(self, value):
        if not isinstance(value, Component):
            raise TypeError(f"'base_configuration' must be of type {Component}, received {type(value)}")
        self._component = value

    def finalize_meshes(self):
        raise NotImplementedError(f"'finalize_meshes' has not been implemented for condition of type {type(self)}")
    
    def assemble_forces_moments(self):
        raise NotImplementedError(f"'assemble_forces_and_moments' has not been implemented for condition of type {type(self)}")