from typing import Union

from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem


class Condition:
    """The Condition class"""
    def __init__(self, fd_axis: Union[Axis, AxisLsdoGeo], control_system: VehicleControlSystem,
                 component: Component) -> None:
        self.parameters: dict = {}
        self.fd_axis = fd_axis
        self.controls = control_system
        self.component = component

    # def assemble_forces_and_moments(self):
    #     # Call solvers:
    #     for a_component_solver in self.component.solvers:
