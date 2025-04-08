from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.vehicle.component import Component
from flight_simulator.core.vehicle.vehicle_control_system import VehicleControlSystem


class Condition:
    """The Condition class"""
    def __init__(self, fd_state: AircraftStates, control_system: VehicleControlSystem, component: Component) -> None:
        self.parameters: dict = {}
        self.state = fd_state
        self.controls = control_system
        self.component = component

    # def assemble_forces_and_moments(self):
    #     # Call solvers:
    #     for a_component_solver in self.component.solvers:
