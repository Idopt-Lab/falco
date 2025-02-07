from pathlib import Path

from flight_simulator.core.loads.mass_properties import MassProperties
import numpy as np
from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from typing import Union
import csdl_alpha as csdl
from dataclasses import dataclass
import warnings
import time
import copy


class ComponentQuantities:
    def __init__(
            self,
            mass_properties: MassProperties = None
    ) -> None:
        """
        Initialize a ComponentQuantities instance.
            surface_area : csdl.Variable
                The computed surface area of the geometry, if applicable.
        """

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
        self.compute_surface_area_flag = compute_surface_area_flag

        # Essential quantities and user-defined parameters
        self.quantities: ComponentQuantities = ComponentQuantities()
        self.parameters: ComponentParameters = ComponentParameters()

        # Store user-defined parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)

        if geometry and compute_surface_area_flag:
            self.quantities.surface_area = self._compute_surface_area(geometry)

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

        # Check if component already exists
        if subcomponent._name in self.comps:
            raise KeyError(f"Subcomponent with name '{subcomponent._name}' already exists.")

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
                self.comps.pop(name)
                subcomponent.parent = None
                return
        raise KeyError(f"Subcomponent {subcomponent._name} not found.")

    def visualize_component_hierarchy(
            self,
            file_name: str = "component_hierarchy",
            file_format: str = "png",
            filepath: Path = Path.cwd(),
            show: bool = False,
    ):
        csdl.check_parameter(file_name, "file_name", types=str)
        csdl.check_parameter(file_format, "file_format", values=("png", "pdf"))
        csdl.check_parameter(show, "show", types=bool)
        try:
            from graphviz import Graph
        except ImportError:
            raise ImportError("Must install graphviz. Can do 'pip install graphviz'")

        # make graph object
        graph = Graph(comment="Component Hierarchy")

        # Go through component hierarchy and components to nodes
        def add_component_to_graph(comp: Component, comp_name: str, parent_name=None):
            graph.node(comp_name, comp_name)
            if parent_name is not None:
                graph.edge(parent_name, comp_name)
            for child_name, child in comp.comps.items():
                add_component_to_graph(child, child_name, comp_name)

        add_component_to_graph(self, self._name)
        graph.render("component_hierarchy",
                     directory=filepath,
                     format=file_format, view=show)

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
    
    
    def connect_component_geometries(
            self, 
            comp_1: "Component", 
            comp_2: "Component",
            connection_point: Union[csdl.Variable, np.ndarray, None] = None):
        
        csdl.check_parameter(comp_1, "comp_1", types=Component)
        csdl.check_parameter(comp_2, "comp_2", types=Component)
        csdl.check_parameter(connection_point, "connection_point", 
                             types=(csdl.Variable, np.ndarray), allow_none=True)
        
        self._geometric_connections = []

        if comp_1.geometry is None:
            raise Exception(f"Comp {comp_1._name} does not have a geometry.")
        if comp_2.geometry is None:
            raise Exception(f"Comp {comp_2._name} does not have a geometry.")

        if connection_point is not None:
            try:
                connection_point.reshape((3,))
            except:
                raise Exception(f"'connection_point' must be of shape (3, ) or reshapable to (3, ). Received shape {connection_point.shape}")

            projection_1 = comp_1.geometry.project(connection_point)
            projection_2 = comp_2.geometry.project(connection_point)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2))
            
        # Else choose the center points of the FFD block
        else:
            point_1 = comp_1.ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            point_2 = comp_2.ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))

            projection_1 = comp_1.geometry.project(point_1)
            projection_2 = comp_2.geometry.project(point_2)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2))

        return

    def setup_geometry(self, plot: bool = False):
        self._geometry_setup_has_been_called = True

        from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()
        system_geometry = self.system.geometry

        if system_geometry is None:
            raise TypeError("'setup_geometry' cannot be called because the geometry associated with the system component is None")

        if not isinstance(system_geometry, FunctionSet):
            raise TypeError(f"The geometry of the system must be of type {FunctionSet}. Received {type(system_geometry)}")

        def setup_geometries(component: "Component"):
            if component.geometry is not None:
                if component._skip_ffd is True:
                    pass
                else:
                    try:
                        component._setup_geometry(parameterization_solver, ffd_geometric_variables, plot=plot)
                    except NotImplementedError:
                        warnings.warn(f"'_setup_geometry' has not been implemented for component {component._name} of {type(component)}")

            if component.comps:
                for comp in component.comps.values():
                    setup_geometries(comp)

        setup_geometries(self.system)

        for projection_1, projection_2, comp_1, comp_2 in self._geometric_connections:
            if isinstance(projection_1, list):
                connection = comp_1.geometry.evaluate(parametric_coordinates=projection_1) - comp_2.geometry.evaluate(parametric_coordinates=projection_2)
            elif isinstance(projection_1, np.ndarray):
                connection = comp_1.ffd_block.evaluate(parametric_coordinates=projection_1) - comp_2.ffd_block.evaluate(parametric_coordinates=projection_2)
            else:
                raise NotImplementedError(f"Unsupported projection type {type(projection_1)}")

            ffd_geometric_variables.add_variable(connection, connection.value)

        t1 = time.time()
        parameterization_solver.evaluate(ffd_geometric_variables)
        t2 = time.time()
        print(f"Time for inner optimization: {t2 - t1}")

        if plot:
            system_geometry.plot(show=True)

