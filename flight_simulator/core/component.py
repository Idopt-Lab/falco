from __future__ import annotations
from pathlib import Path
from flight_simulator.core.loads.mass_properties import MassProperties
from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from typing import Union, List
import numpy as np
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from dataclasses import dataclass
import time
import warnings
import copy


class ComponentQuantities:
    def __init__(self, mass_properties: MassProperties = None) -> None:
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
    def __init__(self, name: str, geometry: Union[FunctionSet, None] = None, compute_surface_area_flag: bool = False, **kwargs) -> None:
        self._name = name
        self.parent = None
        self.comps = {}
        self.geometry: Union[FunctionSet, Geometry, None] = geometry
        self.compute_surface_area_flag = compute_surface_area_flag
        self.quantities: ComponentQuantities = ComponentQuantities()
        self.parameters: ComponentParameters = ComponentParameters()

        for key, value in kwargs.items():
            setattr(self.parameters, key, value)

        if geometry and compute_surface_area_flag:
            self.quantities.surface_area = self._compute_surface_area(geometry)

        if geometry is not None and isinstance(geometry, FunctionSet):
            if "do_not_remake_ffd_block" not in kwargs:
                self._ffd_block = self._make_ffd_block(self.geometry)

    def add_subcomponent(self, subcomponent: Component):
        if not isinstance(subcomponent, Component):
            raise TypeError(f"Subcomponent must be of type 'Component'. Received: {type(subcomponent)}")

        if subcomponent._name in self.comps:
            raise KeyError(f"Subcomponent with name '{subcomponent._name}' already exists.")

        self.comps[subcomponent._name] = subcomponent
        subcomponent.parent = self

    def remove_subcomponent(self, subcomponent: Component):
        for name, comp in self.comps.items():
            if comp == subcomponent:
                self.comps.pop(name)
                subcomponent.parent = None
                return
        raise KeyError(f"Subcomponent {subcomponent._name} not found.")

    def visualize_component_hierarchy(self, file_name: str = "component_hierarchy", file_format: str = "png", filepath: Path = Path.cwd(), show: bool = False):
        csdl.check_parameter(file_name, "file_name", types=str)
        csdl.check_parameter(file_format, "file_format", values=("png", "pdf"))
        csdl.check_parameter(show, "show", types=bool)
        try:
            from graphviz import Graph
        except ImportError:
            raise ImportError("Must install graphviz. Can do 'pip install graphviz'")

        graph = Graph(comment="Component Hierarchy")

        def add_component_to_graph(comp: Component, comp_name: str, parent_name=None):
            graph.node(comp_name, comp_name)
            if parent_name is not None:
                graph.edge(parent_name, comp_name)
            for child_name, child in comp.comps.items():
                add_component_to_graph(child, child_name, comp_name)

        add_component_to_graph(self, self._name)
        graph.render(file_name, directory=filepath, format=file_format, view=show)

    def _make_ffd_block(self, entities, num_coefficients: tuple = (2, 2, 2), order: tuple = (1, 1, 1), num_physical_dimensions: int = 3):
        ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities, num_coefficients=num_coefficients, degree=order)
        ffd_block.coefficients.name = f'{self._name}_coefficients'
        return ffd_block

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot: bool = False):
        rigid_body_translation = csdl.ImplicitVariable(shape=(3,), value=0., name=f'{self._name}_rigid_body_translation')

        for function in self.geometry.functions.values():
            if function.name == "rigid_body_translation":
                shape = function.coefficients.shape
                function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action="k->ijk")

        parameterization_solver.add_parameter(rigid_body_translation, cost=0.1)

    def _compute_surface_area(self, geometry: Geometry, plot_flag: bool = False):
        parametric_mesh_grid_num = 10
        surfaces = geometry.functions
        surface_area = csdl.Variable(shape=(1,), value=1)
        surface_mesh = self.quantities.surface_mesh
        num_surfaces = len(surfaces.keys())

        parametric_mesh = geometry.generate_parametric_grid(grid_resolution=(parametric_mesh_grid_num, parametric_mesh_grid_num))
        coords_vec = geometry.evaluate(parametric_mesh).reshape((num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num, 3))
        surface_mesh.append(coords_vec)

        if plot_flag:
            self.geometry.plot_meshes(coords_vec.reshape((-1, 3)).value)

        coords_u_end = coords_vec[:, 1:, :, :].reshape((num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num, 3))
        coords_u_start = coords_vec[:, :-1, :, :].reshape((num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num, 3))
        coords_v_end = coords_vec[:, :, 1:, :].reshape((num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num - 1, 3))
        coords_v_start = coords_vec[:, :, :-1, :].reshape((num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num - 1, 3))

        u_vectors = coords_u_end - coords_u_start
        u_vectors_1 = u_vectors[:, :, :-1, :].reshape((num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))
        u_vectors_2 = u_vectors[:, :, 1:, :].reshape((num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))

        v_vectors = coords_v_end - coords_v_start
        v_vectors_1 = v_vectors[:, :-1, :, :].reshape((num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))
        v_vectors_2 = v_vectors[:, 1:, :, :].reshape((num_surfaces, parametric_mesh_grid_num - 1, parametric_mesh_grid_num - 1, 3))

        area_vectors_left_lower = csdl.cross(u_vectors_1, v_vectors_2, axis=3)
        area_vectors_right_upper = csdl.cross(v_vectors_1, u_vectors_2, axis=3)
        area_magnitudes_left_lower = csdl.norm(area_vectors_left_lower, ord=2, axes=(3,))
        area_magnitudes_right_upper = csdl.norm(area_vectors_right_upper, ord=2, axes=(3,))
        area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper) / 2
        wireframe_area = csdl.sum(area_magnitudes)
        surface_area = surface_area + wireframe_area

        return surface_area

    def _find_system_component(self, parent) -> FunctionSet:
        if parent is None:
            return self
        else:
            parent = self.parent
            self._find_system_component(parent)


class Configuration:
    def __init__(self, system: Component) -> None:
        if not isinstance(system, Component):
            raise TypeError(f"system must by of type {Component}")
        if system.geometry is not None and not isinstance(system.geometry, FunctionSet):
            raise TypeError(f"If system geometry is not None, it must be of type '{FunctionSet}', received object of type '{type(system.geometry)}'")
        self.system = system
        self._geometric_connections = []
        self._config_copies: List[self] = []

    def connect_component_geometries(self, comp_1: Component, comp_2: Component, connection_point: Union[csdl.Variable, np.ndarray, None] = None, desired_value: Union[csdl.Variable, None] = None):
        csdl.check_parameter(comp_1, "comp_1", types=Component)
        csdl.check_parameter(comp_2, "comp_2", types=Component)
        csdl.check_parameter(connection_point, "connection_point", types=(csdl.Variable, np.ndarray), allow_none=True)

        if comp_1.geometry is None:
            raise Exception(f"Comp {comp_1._name} does not have a geometry.")
        if comp_2.geometry is None:
            raise Exception(f"Comp {comp_2._name} does not have a geometry.")

        if connection_point is not None:
            try:
                connection_point.reshape((3,))
            except:
                raise Exception(f"'connection_point' must be of shape (3,) or reshapable to (3,). Received shape {connection_point.shape}")

            projection_1 = comp_1.geometry.project(connection_point)
            projection_2 = comp_2.geometry.project(connection_point)
            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2, desired_value))
        else:
            point_1 = comp_1._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            point_2 = comp_2._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            projection_1 = comp_1.geometry.project(point_1)
            projection_2 = comp_2.geometry.project(point_2)
            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2, desired_value))

    def setup_geometry(self, additional_constraints: List[tuple] = None, run_ffd: bool = True, plot: bool = False, plot_parameters: dict = None, recorder: csdl.Recorder = None):
        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()
        system_geometry = self.system.geometry

        if system_geometry is None:
            subcomponent_geometries = self._collect_geometries(self.system)
            if subcomponent_geometries:
                system_geometry = FunctionSet(subcomponent_geometries)
                self.system.geometry = system_geometry
            else:
                raise TypeError("'setup_geometry' cannot be called because the geometry associated with the system component is None")

        if not isinstance(system_geometry, FunctionSet):
            raise TypeError(f"The geometry of the system must be of type {FunctionSet}. Received {type(system_geometry)}")

        self._setup_geometries(self.system, parameterization_solver, ffd_geometric_variables, plot)
        self._process_ffd_geometric_variables(ffd_geometric_variables)
        self._process_geometric_connections(ffd_geometric_variables)

        if additional_constraints:
            for constr in additional_constraints:
                connection = csdl.norm(self.system.geometry.evaluate(parametric_coordinates=constr[0]) - self.system.geometry.evaluate(parametric_coordinates=constr[1]))
                ffd_geometric_variables.add_variable(connection, constr[2])

        t1 = time.time()
        if recorder is not None:
            recorder.inline = False
        parameterization_solver.evaluate(ffd_geometric_variables)
        t2 = time.time()

        evaluated_variables = {var_name: var_value for var_name, var_value in ffd_geometric_variables.__dict__.items()}
        self._update_component_geometries(self.system, evaluated_variables)
        print(f"System parameters after update: {self.system.parameters.__dict__}")


        if plot:
            if plot_parameters:
                print(f"System geometry parameters: {self.system.parameters.__dict__}")
                # print(f"System geometry functions: {self.system.geometry.functions}")
                system_geometry.plot(show=True, **plot_parameters)
            else:
                print(f"System geometry parameters: {self.system.parameters.__dict__}")
                # print(f"System geometry functions: {self.system.geometry.functions}")
                system_geometry.plot(show=True)

    def _collect_geometries(self, component):
        geometries = []
        if component.geometry is not None:
            geometries.append(component.geometry)
        for comp in component.comps.values():
            geometries.extend(self._collect_geometries(comp))
        return geometries

    def _setup_geometries(self, component: Component, parameterization_solver, ffd_geometric_variables, plot: bool):
        if component.geometry is not None:
            if not getattr(component, '_skip_ffd', False):
                try:
                    component._setup_geometry(parameterization_solver, ffd_geometric_variables, plot=plot)
                except NotImplementedError:
                    warnings.warn(f"'_setup_geometry' has not been implemented for component {component._name} of {type(component)}")

        for comp in component.comps.values():
            self._setup_geometries(comp, parameterization_solver, ffd_geometric_variables, plot)

    def _process_ffd_geometric_variables(self, ffd_geometric_variables):
        for component in self.system.comps.values():
            if hasattr(component, 'ffd_geometric_variables'):
                computed_values = component.ffd_geometric_variables.computed_values
                desired_values = component.ffd_geometric_variables.desired_values
                chunk_size = 100
                for i in range(0, len(computed_values), chunk_size):
                    chunk_computed_values = computed_values[i:i + chunk_size]
                    chunk_desired_values = desired_values[i:i + chunk_size]
                    for computed_value, desired_value in zip(chunk_computed_values, chunk_desired_values):
                        ffd_geometric_variables.add_variable(computed_value, desired_value)

    def _process_geometric_connections(self, ffd_geometric_variables):
        for connection in self._geometric_connections:
            projection_1, projection_2, comp_1, comp_2, desired_value = connection
            if isinstance(projection_1, list):
                connection = comp_1.geometry.evaluate(parametric_coordinates=projection_1) - comp_2.geometry.evaluate(parametric_coordinates=projection_2)
            elif isinstance(projection_1, np.ndarray):
                connection = comp_1._ffd_block.evaluate(parametric_coordinates=projection_1) - comp_2._ffd_block.evaluate(parametric_coordinates=projection_2)
            else:
                raise NotImplementedError(f"wrong type {type(projection_1)} for projection")

            if desired_value is None:
                ffd_geometric_variables.add_variable(connection, connection.value)
            else:
                ffd_geometric_variables.add_variable(connection, desired_value)


    def _update_component_geometries(self, component, evaluated_vars):
        if component.geometry is not None:
            print(f'var names: {evaluated_vars.keys()}')
            print(f'component parameters: {component.parameters.__dict__}')
            
            desired_values_dict = {}
            computed_values_dict = {}
            
            if 'desired_values' in evaluated_vars:
                for variable in evaluated_vars['desired_values']:
                    if hasattr(variable, 'name') and variable.name is not None:
                        param_name = str(variable.name)
                        value = variable.value if isinstance(variable.value, (int, float, np.ndarray)) else variable.value.value
                        desired_values_dict[param_name] = value

            if 'computed_values' in evaluated_vars:
                for variable in evaluated_vars['computed_values']:
                    if hasattr(variable, 'name') and variable.name is not None:
                        param_name = str(variable.name)
                        value = variable.value if isinstance(variable.value, (int, float, np.ndarray)) else variable.value.value
                        computed_values_dict[param_name] = value
            
            for param_name in component.parameters.__dict__.keys():
                if param_name in desired_values_dict:
                    setattr(component.parameters, param_name, desired_values_dict[param_name])
                    print(f'Updated component.parameters with desired value: {component.parameters}')
                    print(f'param_name: {param_name}')
                    print(f'variable.value: {desired_values_dict[param_name]}')
                elif param_name in computed_values_dict:
                    setattr(component.parameters, param_name, computed_values_dict[param_name])
                    print(f'Updated component.parameters with computed value: {component.parameters}')
                    print(f'param_name: {param_name}')
                    print(f'variable.value: {computed_values_dict[param_name]}')
            
            print(f'component parameters after update: {component.parameters.__dict__}')



        for subcomponent in component.comps.values():
            print(f'Updating subcomponent: {subcomponent._name}')
            print(f'Subcomponent parameters before update: {subcomponent.parameters.__dict__}')
            self._update_component_geometries(subcomponent, evaluated_vars)
            print(f'Subcomponent parameters after update: {subcomponent.parameters.__dict__}')
            
            # Propagate subcomponent parameters to the system parameters
            for param_name, param_value in subcomponent.parameters.__dict__.items():
                unique_param_name = f'{subcomponent._name}.{param_name}'
                if unique_param_name not in component.parameters.__dict__:
                    setattr(component.parameters, unique_param_name, param_value)
                    print(f'Propagated {unique_param_name} from subcomponent {subcomponent._name} to system component')
                else:
                    # Handle the case where the parameter already exists in the system parameters
                    # You can choose to overwrite or merge the values as needed
                    print(f'Parameter {unique_param_name} already exists in system component, skipping propagation')
        
        print(f'System component parameters after propagating subcomponent parameters: {component.parameters.__dict__}')