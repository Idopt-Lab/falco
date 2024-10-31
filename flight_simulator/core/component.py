from flight_simulator.utils.mass_properties import MassProperties


from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from typing import Union, List
import numpy as np
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from dataclasses import dataclass
import time



class ComponentQuantities:
    def __init__(
            self,
            mass_properties: MassProperties = None,
    ) -> None:

        self._mass_properties = mass_properties

        self.surface_mesh = []
        self.surface_area = None

        if mass_properties is None:
            self.mass_properties = MassProperties()

    @property
    def mass_properties(self):
        return self._mass_properties

    @mass_properties.setter
    def mass_properties(self, value):
        if not isinstance(value, MassProperties):
            raise ValueError(f"'mass_properties' must be of type {MassProperties}, received {value}")

        self._mass_properties = value


@dataclass
class ComponentParameters:
    pass


class Component:
    """The base component class.

    Attributes
    ---------
    comps : dictionary
        Dictionary of the sub/children components

    geometry : Union[Geometry]

    quantities : dictionary
        General container data; by default contains
        - mass_properties
    """

    # Instance counter for naming components under the hood
    _instance_count = 0

    parent = None

    def __init__(self, geometry: Union[FunctionSet, None] = None,
                 **kwargs) -> None:
        csdl.check_parameter(geometry, "geometry", types=(FunctionSet), allow_none=True)

        # Increment instance count and set private component name (will be obsolete in the future)
        Component._instance_count += 1
        self._name = f"component_{self._instance_count}"

        # set class attributes
        self.geometry: Union[FunctionSet, Geometry, None] = geometry
        self.comps: ComponentDict = ComponentDict(parent=self)
        self.quantities: ComponentQuantities = ComponentQuantities()
        self.parameters: ComponentParameters = ComponentParameters()

        # Set any keyword arguments on parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)


class ComponentDict(dict):
    def __init__(self, parent: Component, *args, **kwargs):
        super().__init__(*args, **kwargs)
        csdl.check_parameter(parent, "parent", types=(Component))
        self.parent = parent

    def __getitem__(self, key) -> Component:
        # Check if key exists
        if key not in self:
            raise KeyError(f"The component '{key}' does not exist. Existing components: {list(self.keys())}")
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value: Component, allow_overwrite=False):
        # Check type
        if not isinstance(value, Component):
            raise TypeError(f"Components must be of type(s) {Component}; received {type(value)}")

        # Check if key is already specified
        elif key in self:
            if allow_overwrite is False:
                raise Exception(f"Component {key} has already been set and cannot be re-set.")
            else:
                super().__setitem__(key, value)
                value.parent = self.parent

        # Set item otherwise
        else:
            super().__setitem__(key, value)
            value.parent = self.parent