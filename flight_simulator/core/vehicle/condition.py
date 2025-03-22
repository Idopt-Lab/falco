from flight_simulator.core.vehicle.component import Component


class Condition:
    """The Condition class"""
    def __init__(self) -> None:
        self.parameters : dict = {}
        self._component : Component = None

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