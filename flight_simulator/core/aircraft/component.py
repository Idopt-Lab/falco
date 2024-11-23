from flight_simulator.core.loads.mass_properties import MassProperties


from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from typing import Union
import csdl_alpha as csdl
from dataclasses import dataclass


class ComponentQuantities:
    def __init__(
            self,
            mass_properties: MassProperties = None
    ) -> None:

        self._mass_properties = mass_properties

    @property
    def mass_properties(self):
        return self._mass_properties

    @mass_properties.setter
    def mass_properties(self, value):
        if not isinstance(value, MassProperties):
            raise ValueError(f"'mass_properties' must be of type {MassProperties}, received {type(value)}")
        self._mass_properties = value


@dataclass
class ComponentParameters:
    pass


class Component:
    """
    The base component class.

    Attributes
    ----------
    name : str
        A unique name for the component.

    parent : Component
        The parent component, if the current component is part of a hierarchy.

    comps : dict
        A dictionary of subcomponents with their names as keys.

    geometry : FunctionSet or None
        The geometry associated with the component, enabling geometric operations.

    quantities : ComponentQuantities
        Essential quantities a component must always have, such as mass properties.

    parameters : ComponentParameters
        A container for user-defined parameters, initialized using keyword arguments.

    compute_surface_area : bool
        Indicates whether to compute the surface area of the geometry upon initialization.

    surface_area : csdl.Variable
        The computed surface area of the geometry, if applicable.
    """

    def __init__(self,
                 name: str,
                 geometry: Union[FunctionSet, None] = None,
                 compute_surface_area_flag: bool = False,
                 **kwargs) -> None:
        """
        Initialize a Component instance.

        Parameters
        ----------
        geometry : FunctionSet or None, optional
            The geometry associated with the component.

        name : str
            A custom name for the component. If not provided, a unique name will be generated.

        compute_surface_area : bool, optional
            Whether to compute the surface area of the geometry upon initialization. Default is True.

        **kwargs : dict
            User-defined parameters to be stored in the `parameters` attribute.
        """
        # Increment instance count and set the name
        self._name = name

        # Hierarchical attributes
        self.parent = None
        self.comps = {}  # Dictionary to hold subcomponents

        # Geometry-related attributes
        self.geometry: Union[FunctionSet, Geometry, None] = geometry
        self.surface_area = None
        self.compute_surface_area_flag = compute_surface_area_flag

        if geometry and compute_surface_area_flag:
            self.surface_area = self._compute_surface_area(geometry)

        # Essential quantities and user-defined parameters
        self.quantities: ComponentQuantities = ComponentQuantities()
        self.parameters: ComponentParameters = ComponentParameters()

        # Store user-defined parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)