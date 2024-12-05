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

        self.surface_mesh = []
        self.surface_area = None

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

        # Essential quantities and user-defined parameters
        self.quantities: ComponentQuantities = ComponentQuantities()
        self.parameters: ComponentParameters = ComponentParameters()

        # Store user-defined parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)

        if geometry and compute_surface_area_flag:
            self.surface_area = self._compute_surface_area(geometry)

    def add_subcomponent(self, subcomponent):
        """
        Add a subcomponent to the current component.

        Parameters
        ----------
        subcomponent : Component
            The subcomponent to add.

        Raises
        ------
        TypeError
            If the provided subcomponent is not an instance of Component.
        """
        if not isinstance(subcomponent, Component):
            raise TypeError(f"Subcomponent must be of type 'Component'. Received: {type(subcomponent)}")

        self.comps[subcomponent._name] = subcomponent
        subcomponent.parent = self

    def remove_subcomponent(self, subcomponent):
        """
        Remove a subcomponent from the current component.

        Parameters
        ----------
        subcomponent : Component
            The subcomponent to remove.

        Raises
        ------
        KeyError
            If the subcomponent is not found in the current component's hierarchy.
        """
        for name, comp in self.comps.items():
            if comp == subcomponent:
                del self.comps[name]
                subcomponent.parent = None
                return
        raise KeyError(f"Subcomponent {subcomponent._name} not found.")

    def _compute_surface_area(self, geometry: Geometry,
                              plot_flag: bool = False):
        """Compute the surface area of a component."""
        parametric_mesh_grid_num = 10

        surfaces = geometry.functions
        surface_area = csdl.Variable(shape=(1,), value=1)

        surface_mesh = self.quantities.surface_mesh

        num_surfaces = len(surfaces.keys())

        parametric_mesh = geometry.generate_parametric_grid(
            grid_resolution=(parametric_mesh_grid_num, parametric_mesh_grid_num))
        coords_vec = geometry.evaluate(parametric_mesh).reshape(
            (num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num, 3))
        surface_mesh.append(coords_vec)

        if plot_flag:
            self.geometry.plot_meshes(coords_vec.reshape((-1, 3)).value)

        coords_u_end = coords_vec[:, 1:, :, :].reshape(
            (num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num, 3))
        coords_u_start = coords_vec[:, :-1, :, :].reshape(
            (num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num, 3))

        coords_v_end = coords_vec[:, :, 1:, :].reshape(
            (num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num - 1, 3))
        coords_v_start = coords_vec[:, :, :-1, :].reshape(
            (num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num - 1, 3))

        u_vectors = coords_u_end - coords_u_start
        u_vectors_start = u_vectors  # .reshape((-1, ))
        u_vectors_1 = u_vectors_start[:, :, :-1, :].reshape(
            (num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))
        u_vectors_2 = u_vectors_start[:, :, 1:, :].reshape(
            (num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))

        v_vectors = coords_v_end - coords_v_start
        v_vectors_start = v_vectors  # .reshape((-1, ))
        v_vectors_1 = v_vectors_start[:, :-1, :, :].reshape(
            (num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))
        v_vectors_2 = v_vectors_start[:, 1:, :, :].reshape(
            (num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))

        area_vectors_left_lower = csdl.cross(u_vectors_1, v_vectors_2, axis=3)
        area_vectors_right_upper = csdl.cross(v_vectors_1, u_vectors_2, axis=3)
        area_magnitudes_left_lower = csdl.norm(area_vectors_left_lower, ord=2, axes=(3,))
        area_magnitudes_right_upper = csdl.norm(area_vectors_right_upper, ord=2, axes=(3,))
        area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper) / 2
        wireframe_area = csdl.sum(area_magnitudes)
        surface_area = surface_area + wireframe_area

        return surface_area
