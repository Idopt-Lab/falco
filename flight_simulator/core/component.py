from __future__ import annotations
from pathlib import Path
from flight_simulator.core.loads.mass_properties import MassProperties
from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from typing import Union, List
import numpy as np
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from dataclasses import dataclass
import time
import warnings
import copy



class VectorizedAttributes:
    def __init__(self, attribute_list, num_nodes) -> None:
        self.attribute_list = attribute_list
        self.num_nodes = num_nodes

    def __getattr__(self, name):
        child_attribute_list = []
        if hasattr(self.attribute_list[0], name):
            if callable(getattr(self.attribute_list[0], name)):
                def method(*args, **kwargs):
                    if 'vectorized' in kwargs:
                        if not kwargs['vectorized']:
                            kwargs.pop('vectorized')
                            return getattr(self.attribute_list[0], name)(*args, **kwargs)
                    return_list = []
                    for comp in self.attribute_list:
                        output = getattr(comp, name)(*args, **kwargs)
                        if output:
                            return_list.append(output)
                    return return_list
                return method
            else:
                for i in range(self.num_nodes):
                    attr = getattr(self.attribute_list[i], name)
                    child_attribute_list.append(attr)
                
                if isinstance(child_attribute_list[0], (list, dict, set)) or hasattr(child_attribute_list[0], '__dict__'):
                    return VectorizedAttributes(child_attribute_list, self.num_nodes)
                else:
                    return child_attribute_list
        else:
            existing_attrs = [attr for attr in dir(self.attribute_list[0]) if not attr.startswith(("__", "_"))]
            raise AttributeError(f"Attribute {name} does not exist. Existing attributes are {existing_attrs}")
        
    def __setattr__(self, name: str, value) -> None:
        if name in {"attribute_list", "num_nodes"}:
            # Directly set the instance attributes
            super().__setattr__(name, value)
        else:
            # Set the attribute on each component in the attribute list
            for comp in self.attribute_list:
                setattr(comp, name, value)



class ComponentQuantities:
    def __init__(
        self, 
        mass_properties: MassProperties = None,
        utils : dict = {},
    ) -> None:
        
        self._mass_properties = mass_properties
        self.utils = utils
    
        self.surface_mesh = []
        self.surface_area = None

        

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
        - function_spaces
        - meshes
    """
    # Default function spaces for components 
    _constant_b_spline_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
    _linear_b_spline_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    _linear_b_spline_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
    _quadratic_b_spline_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(3,))
    _cubic_b_spline_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))
    
    # Instance counter for naming components under the hood
    _instance_count = 0

    # Boolean attribute to keep track of whether component is a copy
    _is_copy = False

    # Private attrbute to allow certain components to be excluded from FFD    
    _skip_ffd = False

    parent = None

    def __init__(self, geometry : Union[FunctionSet, None]=None, 
                 compute_surface_area: bool=True, skip_ffd: bool=False, **kwargs) -> None: 
        csdl.check_parameter(geometry, "geometry", types=(FunctionSet), allow_none=True)
        
        # Increment instance count and set private component name (will be obsolete in the future)
        Component._instance_count += 1
        self._name = f"component_{self._instance_count}"
        self.compute_surface_area = compute_surface_area
        self.skip_ffd = skip_ffd

        # set class attributes
        self.geometry : Union[FunctionSet, Geometry, None] = geometry
        self.comps : ComponentDict = ComponentDict(parent=self)
        self.quantities : ComponentQuantities = ComponentQuantities()
        self.parameters : ComponentParameters = ComponentParameters()

        # Set any keyword arguments on parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)
        
        if geometry is not None and isinstance(geometry, FunctionSet):
            if self.compute_surface_area:
                self.quantities.surface_area = self._compute_surface_area(geometry=geometry)
            if "do_not_remake_ffd_block" in kwargs.keys():
                pass
            else:
                self._ffd_block = self._make_ffd_block(self.geometry)

                self.ffd_block_face_1 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.]))
                self.ffd_block_face_2 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 1.]))
                self.ffd_block_face_3 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0., 0.5]))
                self.ffd_block_face_4 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.5]))
                self.ffd_block_face_5 = self._ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5]))
                self.ffd_block_face_6 = self._ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5]))
                self.ffd_block_center = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
    
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
        
    def plot(self):
        """Plot a component's geometry."""
        if self.geometry is None:
            raise ValueError(f"Cannot plot component {self} since its geometry is None.")
        else:
            self.geometry.plot()

    def actuate(self):
        raise NotImplementedError(f"'actuate' has not been implemented for component of type {type(self)}")

    def _make_ffd_block(self, entities, 
                        num_coefficients : tuple=(2, 2, 2), 
                        order: tuple=(1, 1, 1), 
                        num_physical_dimensions : int=3):
        """
        Call 'construct_ffd_block_around_entities' function.

        This method constructs a Cartesian FFD block with linear B-splines
        and 2 degrees of freedom in all dimensions.
        """
        ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities, 
                                                   num_coefficients=num_coefficients, degree=order)
        ffd_block.coefficients.name = f'{self._name}_coefficients'

        return ffd_block 
    
    def _setup_ffd_block(self):
        raise NotImplementedError(f"'_setup_ffd_block' has not been implemented for {type(self)}")
    
    def _extract_geometric_quantities_from_ffd_block(self):
        raise NotImplementedError(f"'_extract_geometric_quantities_from_ffd_block' has not been implemented for {type(self)}")
    
    def _setup_ffd_parameterization(self):
        raise NotImplementedError(f"'_setup_ffd_parameterization' has not been implemented for {type(self)}")

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot=False, parent_translation=None):
        # Add rigid body translation (without FFD)
        if parent_translation is None:
            rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0., name=f'{self._name}_rigid_body_translation')
        else:
            rigid_body_translation = parent_translation + csdl.ImplicitVariable(shape=(3, ), value=0., name=f'{self._name}_rigid_body_translation')

        for function in self.geometry.functions.values():
            shape = function.coefficients.shape
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action="k->ijk")
        
        parameterization_solver.add_parameter(rigid_body_translation, cost=0.1)
        return rigid_body_translation

    def _find_system_component(self, parent) -> FunctionSet:
        """Find the top-level system component by traversing the component hiearchy"""
        if parent is None:
            return self
        else:
            parent = self.parent
            self._find_system_component(parent)

    def _compute_surface_area(self, geometry:Geometry):
        import time 
        """Compute the surface area of a component."""
        parametric_mesh_grid_num = 15

        surfaces = geometry.functions
        surface_area = csdl.Variable(shape=(1, ), value=1)

        surface_mesh = self.quantities.surface_mesh

        num_surfaces = len(surfaces.keys())

        parametric_mesh = geometry.generate_parametric_grid(grid_resolution=(parametric_mesh_grid_num, parametric_mesh_grid_num))
        coords_vec = geometry.evaluate(parametric_mesh).reshape((num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num, 3))
        surface_mesh.append(coords_vec)


        coords_u_end = coords_vec[:, 1:, :, :].reshape((num_surfaces, parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))
        coords_u_start = coords_vec[:, :-1, :, :].reshape((num_surfaces, parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))

        coords_v_end = coords_vec[:, :, 1:, :].reshape((num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num-1, 3))
        coords_v_start = coords_vec[:, :, :-1, :].reshape((num_surfaces, parametric_mesh_grid_num, parametric_mesh_grid_num-1, 3))

        u_vectors = coords_u_end - coords_u_start
        u_vectors_start = u_vectors # .reshape((-1, ))
        u_vectors_1 = u_vectors_start[:, :, :-1, :].reshape((num_surfaces, parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))
        u_vectors_2 = u_vectors_start[:, :, 1:, :].reshape((num_surfaces, parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))


        v_vectors = coords_v_end - coords_v_start
        v_vectors_start = v_vectors # .reshape((-1, ))
        v_vectors_1 = v_vectors_start[:, :-1, :, :].reshape((num_surfaces, parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))
        v_vectors_2 = v_vectors_start[:, 1:, :, :].reshape((num_surfaces, parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))

        area_vectors_left_lower = csdl.cross(u_vectors_1, v_vectors_2, axis=3)
        area_vectors_right_upper = csdl.cross(v_vectors_1, u_vectors_2, axis=3)
        area_magnitudes_left_lower = csdl.norm(area_vectors_left_lower, ord=2, axes=(3, ))
        area_magnitudes_right_upper = csdl.norm(area_vectors_right_upper, ord=2, axes=(3, ))
        area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper)/2
        wireframe_area = csdl.sum(area_magnitudes)
        surface_area =  surface_area + wireframe_area

        return surface_area


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
    
    def __setitem__(self, key, value : Component, allow_overwrite=False):
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


if __name__ == "__main__":
    def unpack_attributes(obj, _depth=0, _visited=None):
        if _visited is None:
            _visited = set()
            
        obj_id = id(obj)
        if obj_id in _visited:
            return {}
        
        _visited.add(obj_id)
        
        attributes = {}
        for attr_name in dir(obj):
            # Ignore private and protected attributes
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
            except AttributeError:
                continue
            
            # Check if the attribute is itself an object that we should unpack
            if hasattr(attr_value, '__dict__'):
                attributes[attr_name] = unpack_attributes(attr_value, _depth + 1, _visited)
            else:
                attributes[attr_name] = attr_value
        
        return attributes



class Configuration:
    """The configurations class"""
    def __init__(self, system : Component) -> None:
        # Check whether system if a component
        if not isinstance(system, Component):
            raise TypeError(f"system must by of type {Component}")
        # Check that if system geometry is not None, it is of the correct type
        if system.geometry is not None:
            if not isinstance(system.geometry, FunctionSet ):
                raise TypeError(f"If system geometry is not None, it must be of type '{FunctionSet}', received object of type '{type(system.geometry)}'")
        self.system = system
        self._is_copy : bool = False
        self._geometry_setup_has_been_called = False
        self._geometric_connections = []
        self._config_copies: List[self] = []

    def visualize_component_hierarchy(
            self, 
            file_name: str="component_hierarchy", 
            file_format: str="png",
            show: bool=False,
        ):
        csdl.check_parameter(file_name, "file_name", types=str)
        csdl.check_parameter(file_format, "file_format", values=("png", "pdf"))
        csdl.check_parameter(show, "show", types=bool)
        try:
            from graphviz import Graph
        except ImportError:
            raise ImportError("Must install graphviz. Can do 'pip install graphviz'")
        
        # make graph object
        graph = Graph(comment="Compopnent Hierarchy")

        # Go through component hierarchy and components to nodes
        def add_component_to_graph(comp: Component, comp_name: str, parent_name=None):
            graph.node(comp_name, comp_name)
            if parent_name is not None:
                graph.edge(parent_name, comp_name)
            for child_name, child in comp.comps.items():
                add_component_to_graph(child, child_name, comp_name)

        add_component_to_graph(self.system, "system")
        graph.render("component_hierarchy", format=file_format, view=show)

    def assemble_meshes(self):
        """Assemble all component meshes into the mesh container."""
        def add_mesh_to_container(comp: Component):
            for mesh_name, mesh in comp._discretizations.items():
                self.mesh_container[mesh_name] = mesh
            if comp.comps:
                for sub_comp_name, sub_comp in comp.comps.items():
                    add_mesh_to_container(sub_comp)
        
        for comp_name, comp in self.system.comps.items():
            add_mesh_to_container(comp)



    
    def remove_component(self, comp : Component):
        """Remove a component from a configuration."""
        
        # Check that comp is the right type
        if not isinstance(comp, Component):
            raise TypeError(f"Can only remove components. Receieved type {type(comp)}")
        
        # Check if comp is the system itself
        if comp == self.system:
            raise Exception("Cannot remove system component.")
        
        def remove_comp_from_dictionary(comp, comp_dict)-> bool:
            """
            Remove a component from a dictionary.

            Arguments
            ---------
            - comp : original component to be removed

            - comp_dict : component dictionary (will be changed if comp not in comp_dict)

            """
            for sub_comp_name, sub_comp in comp_dict.items():
                if comp == sub_comp:
                    # del comp_dict[sub_comp_name]
                    comp_dict.pop(sub_comp_name)
                    return True
                else:
                    if remove_comp_from_dictionary(comp, sub_comp.comps):
                        return True
            return False
        
        comp_exists = remove_comp_from_dictionary(comp, self.system.comps)

        if comp_exists:
            return
        else:
            raise Exception(f"Cannot remove component {comp._name} of type {type(comp)} from the configuration since it does not exists.")

    def assemble_system_mass_properties(
            self, 
            point : np.ndarray = np.array([0., 0., 0.]),
            update_copies: bool = False
        ):
        """Compute the mass properties of the configuration.
        """
        csdl.check_parameter(update_copies, "update_copies", types=bool)

        system = self.system
        system_comps = system.comps

        if not np.array_equal(point, np.array([0. ,0., 0.])):
            raise NotImplementedError("Mass properties taken w.r.t. a specific point not yet implemented.")

        def _sum_component_masses(
            comps,
            system_mass=0.,
        ):
            """Sum all component mass to compute system mass.

            Parameters
            ----------
            comps : _type_
                dictionary of children components
            system_mass : _type_, optional
                Initial system mass, by default 0.
            """
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")
                    system_mass = system_mass * 1

                # Add component mass to system mass
                else:
                    m = mass_props.mass
                    if m is not None:
                        system_mass = system_mass + m

                # Check if the component has children
                if not comp.comps:
                    pass

                # If comp has children, add their mass recursively 
                else:
                    system_mass = \
                        _sum_component_masses(comp.comps, system_mass)

            return system_mass
        
        def _sum_component_cgs(
            comps,
            system_mass,
            system_cg=np.zeros((3, ))
        ):
            # second, compute total cg 
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_cg = system_cg * 1

                # otherwise add component cg contribution
                else:
                    cg = mass_props.cg_vector
                    m = mass_props.mass
                    if cg is not None:
                        system_cg = system_cg + m * cg / system_mass

                # Check if the component has children
                if not comp.comps:
                    pass

                # If comp has children, add their mass recursively 
                else:
                    system_cg = \
                        _sum_component_cgs(comp.comps, system_mass, system_cg)

            return system_cg
        
        def _sum_component_inertias(
            comps,
            system_cg,
            system_inertia_tensor=np.zeros((3, 3)),
        ):
            # system-level cg
            x_cg_sys = system_cg[0]
            y_cg_sys = system_cg[1]
            z_cg_sys = system_cg[2]

            # Third, compute total cg and inertia tensor
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_inertia_tensor = system_inertia_tensor * 1

                else:
                    it = mass_props.inertia_tensor
                    m = mass_props.mass
                    cg = mass_props.cg_vector

                    if cg is not None:
                        # component-level cg
                        x_cg_comp = cg[0]
                        y_cg_comp = cg[1]
                        z_cg_comp = cg[2]

                        x = x_cg_comp - x_cg_sys
                        y = y_cg_comp - y_cg_sys
                        z = z_cg_comp - z_cg_sys

                    # use given inertia if provided
                    if it is not None:
                        if m is None:
                            raise Exception(f"Component {comp_name}, has an inertia tensor but no mass. Cannot apply parallel axis theorem.")
                        if cg is None:
                            raise Exception(f"Component {comp_name}, has an inertia tensor but no cg_vector. Cannot apply parallel axis theorem.")
                        
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = it[0, 0] + m * (y**2 + z**2)
                        ixy = it[0, 1] - m * (x * y)
                        ixz = it[0, 2] - m * (x * z)
                        iyx = ixy 
                        iyy = it[1, 1] + m * (x**2 + z**2)
                        iyz = it[1, 2] - m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = it[2, 2] + m * (x**2  + y**2)

                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)

                        system_inertia_tensor = system_inertia_tensor + it
                    
                    # point mass assumption
                    elif m is not None: 
                        if cg is None:
                            raise Exception(f"Component {comp_name}, has a specified mass no cg_vector. Cannot apply parallel axis theorem to sum up system-level mass properties.")

                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = m * (y**2 + z**2)
                        ixy = -m * (x * y)
                        ixz = -m * (x * z)
                        iyx = ixy 
                        iyy = m * (x**2 + z**2)
                        iyz = -m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = m * (x**2  + y**2)
                        
                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)
                        
                        system_inertia_tensor = system_inertia_tensor + it

                # Check if the component has children
                if not comp.comps:
                    pass

                # If comp has children, add their mass recursively 
                else:
                    system_inertia_tensor = \
                        _sum_component_inertias(comp.comps, system_cg, system_inertia_tensor)

            return system_inertia_tensor

        def _add_component_mps_to_system_mps(
                comps, 
                system_mass=0., 
                system_cg=np.zeros((3, )), 
                system_inertia_tensor=np.zeros((3, 3))
            ):
            """Add component-level mass properties to the system-level mass properties. 
            Only called internally by 'assemble_system_mass_properties'"""
            # first, compute total mass
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")

                    system_mass = system_mass * 1

                else:
                    m = mass_props.mass
                    
                    if m is not None:
                        system_mass = system_mass + m
        
            # second, compute total cg 
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_cg = system_cg * 1
                    system_inertia_tensor = system_inertia_tensor * 1

                # otherwise add component cg contribution
                else:
                    cg = mass_props.cg_vector
                    m = mass_props.mass
                    if cg is not None:
                        system_cg = system_cg + m * cg / system_mass
           
            # Third, compute total cg and inertia tensor
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_inertia_tensor = system_inertia_tensor * 1

                else:
                    it = mass_props.inertia_tensor
                    m = mass_props.mass

                    x = system_cg[0]
                    y = system_cg[1]
                    z = system_cg[2]
                    
                    # use given mps
                    if it is not None:
                        if m is None:
                            raise Exception(f"Component {comp_name}, has an inertia tensor but no mass. Cannot apply parallel axis theorem.")
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = it[0, 0] + m * (y**2 + z**2)
                        ixy = it[0, 1] - m * (x * y)
                        ixz = it[0, 2] - m * (x * z)
                        iyx = ixy 
                        iyy = it[1, 1] + m * (x**2 + z**2)
                        iyz = it[1, 2] - m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = it[2, 2] + m * (x**2  + y**2)

                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)

                        system_inertia_tensor = system_inertia_tensor + it
                    
                    # point mass assumption
                    elif m is not None: 
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = m * (y**2 + z**2)
                        ixy = -m * (x * y)
                        ixz = -m * (x * z)
                        iyx = ixy 
                        iyy = m * (x**2 + z**2)
                        iyz = -m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = m * (x**2  + y**2)
                        
                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)
                        
                        system_inertia_tensor = system_inertia_tensor + it
                
                # Check if the component has children
                if not comp.comps:
                    pass

                # If their children, add their mass properties via a private method
                else:
                    system_mass, system_cg, system_inertia_tensor = \
                        _add_component_mps_to_system_mps(comp.comps, system_mass, system_cg, system_inertia_tensor)

            return system_mass, system_cg, system_inertia_tensor

        # Check if mass properties of system has already been set/computed
        # 1) masss, cg, and inertia tensor have all been defined
        system_mps = system.quantities.mass_properties
        if all(getattr(system_mps, mp) is not None for mp in system_mps.__dict__):
            # Check if the system is a copy
            if system._is_copy:
                system_mass = _sum_component_masses(system_comps)
                system_cg = _sum_component_cgs(system_comps, system_mass=system_mass)
                system_inertia_tensor = _sum_component_inertias(system_comps, system_cg=system_cg)
                
                # system_mass, system_cg, system_inertia_tensor = \
                # _add_component_mps_to_system_mps(system_comps)

                system.quantities.mass_properties.mass = system_mass
                system.quantities.mass_properties.cg_vector = system_cg
                system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

            else:
                warnings.warn(f"System already has defined mass properties: {system_mps}")
                return

        # 2) mass and cg have been defined and inertia tensor is None
        elif system_mps.mass is not None and system_mps.cg_vector is not None and system_mps.inertia_tensor is None:
            system_inertia_tensor = np.zeros((3, 3))
            warnings.warn(f"System already has defined mass and cg vector; will compute inertia tensor based on point mass assumption")
            x = system_mps.cg_vector[0]
            y = system_mps.cg_vector[1]
            z = system_mps.cg_vector[2]
            m = system_mps.mass
            ixx = m * (y**2 + z**2)
            ixy = -m * (x * y)
            ixz = -m * (x * z)
            iyx = ixy 
            iyy = m * (x**2 + z**2)
            iyz = -m * (y * z)
            izx = ixz 
            izy = iyz 
            izz = m * (x**2  + y**2)
            system_mps.inertia_tensor = np.array([
                [ixx, ixy, ixz],
                [iyx, iyy, iyz],
                [izx, izy, izz],
            ])
            return 
        
        # 3) only mass has been defined
        elif system_mps.mass is not None and system_mps.cg_vector is None and system_mps.inertia_tensor is None:
            raise Exception("Partially defined system mass properties; only system mass has been set. Need at least mass and the cg vector.")
        
        # 4) only cg vector has been defined
        elif system_mps.mass is None and system_mps.cg_vector is not None and system_mps.inertia_tensor is None:
            raise Exception("Partially defined system mass properties; only system cg_vector has been set. Need at least mass and the cg vector.")

        # 5) only inertia tensor vector has been defined
        elif system_mps.mass is None and system_mps.cg_vector is None and system_mps.inertia_tensor is not None:
            raise Exception("Partially defined system mass properties; only system inertia_tensor has been set. Need to also specify mass and cg vector.")

        # 6) Inertia tensor is not None and cg vector is not None
        elif system_mps.mass is None and system_mps.cg_vector is not None and system_mps.inertia_tensor is not None:
            raise Exception("Partially defined system mass properties; only system cg_vector and inertia_tensor has been set. Mass has not been set.")

        else:
            # check if system has any components
            if not system_comps:
                raise Exception("System does not have any subcomponents and does not have any mass properties. Cannot assemble mass properties.")
            
            # loop over all components and sum the mass properties
            system_mass = 0
            system_cg = np.array([0., 0., 0.])
            system_inertia_tensor = np.zeros((3, 3))

            system_mass = _sum_component_masses(system_comps,system_mass=system_mass)
            system_cg = _sum_component_cgs(system_comps, system_mass=system_mass, system_cg=system_cg)
            system_inertia_tensor = _sum_component_inertias(system_comps, system_cg=system_cg, system_inertia_tensor=system_inertia_tensor)
    
            # system_mass, system_cg, system_inertia_tensor = \
            #     _add_component_mps_to_system_mps(system_comps, system_mass, system_cg, system_inertia_tensor)

            system.quantities.mass_properties.mass = system_mass
            system.quantities.mass_properties.cg_vector = system_cg
            system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

        # Update mass properties for any copied configurations
        if update_copies:

            def _update_comp_copy_mps(component_copy: Component, original_component: Component):
                # Get original mass properties
                original_mps = original_component.quantities.mass_properties

                # Copy original mass properties and set them as new ones
                component_copy.quantities.mass_properties = copy.copy(original_mps)

                for child_name, child_copy in component_copy.comps.items():
                    if child_name in original_component.comps.keys():
                        original_child = original_component.comps[child_name]
                        _update_comp_copy_mps(child_copy, original_child)

            def _update_config_copy_mps(config_copy: self):
                # print("Config copy type", type(config_copy))
                config_copy_system = config_copy.system
                original_system = system
                _update_comp_copy_mps(config_copy_system, original_system)


            for config_copy in self._config_copies:
                _update_config_copy_mps(config_copy)

                # _print_existing_mps(config_copy.system)

    def connect_component_geometries(
        self,
        comp_1: Component,
        comp_2: Component,
        connection_point: Union[csdl.Variable, np.ndarray, None]=None,
        desired_value: Union[csdl.Variable, None]=None
    ):
        """Connect the geometries of two components.

        Function to ensure a component geometries can rigidly 
        translate if the component it is connected to moves.

        Parameters
        ----------
        comp_1 : Component
            the first component to be connected
        comp_2 : Component
            the second component to be connected
        connection_point : Union[csdl.Variable, np.ndarray, None], optional
            the point with respect to which the connection is defined. E.g., 
            if the wing and fuselage geometries are connected, this point 
            could be the quarter chord of the wing. This means that the distance
            between the point and the two component will remain constant, 
            by default None
        desired_value : Union[csdl.Variable, np.ndarray, None], optional
            The value to be enforced by the inner optimization, if None,
            the connection point's initial value will be chosen

        Raises
        ------
        Exception
            If 'comp_1' is not an instances of Compone
        Exception
            If 'comp_2' is not an instances of Compone
        Exception
            If 'connection_point' is not of shape (3, )
        """
        csdl.check_parameter(comp_1, "comp_1", types=Component)
        csdl.check_parameter(comp_2, "comp_2", types=Component)
        csdl.check_parameter(connection_point, "connection_point" ,
                             types=(csdl.Variable, np.ndarray), allow_none=True)

        # Check that comp_1 and comp_2 have geometries
        if comp_1.geometry is None:
            raise Exception(f"Comp {comp_1.name} does not have a geometry.")
        if comp_2.geometry is None:
            raise Exception(f"Comp {comp_2.name} does not have a geometry.")
        
        # If connection point provided, check that its shape is (3, )
        if connection_point is not None:
            try:
                connection_point.reshape((3, ))
            except:
                raise Exception(f"'connection_point' must be of shape (3, ) or reshapable to (3, ). Received shape {connection_point.shape}")

            projection_1 = comp_1.geometry.project(connection_point)
            projection_2 = comp_2.geometry.project(connection_point)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2, desired_value))
        
        # Else choose the center points of the FFD block
        else:
            point_1 = comp_1._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            point_2 = comp_2._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))

            projection_1 = comp_1.geometry.project(point_1)
            projection_2 = comp_2.geometry.project(point_2)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2, desired_value))
        
        return

    def setup_geometry(self, additional_constraints: List[tuple]=None, run_ffd : bool=True, plot : bool=False, recorder: csdl.Recorder =None):
        """Run the geometry parameterization solver. 
        
        Note: This is only allowed on the based configuration.
        """
        self._geometry_setup_has_been_called = True

        if self._is_copy and run_ffd:
            raise Exception("With curent version of CADDEE, Cannot call setup_geometry with run_ffd=True on a copy of a configuration.")
        
        from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()
        system_geometry = self.system.geometry

        if system_geometry is None:
            raise TypeError("'setup_geometry' cannot be called because the geometry asssociated with the system component is None")

        if not isinstance(system_geometry, FunctionSet):
            raise TypeError(f"The geometry of the system must be of type {FunctionSet}. Received {type(system_geometry)}")

        def setup_geometries(component: Component, parent_translation=None):
            # If component has a geometry, set up its geometry
            if component.geometry is not None:
                if component._skip_ffd is True:
                    pass
                else:
                    try: # NOTE: might cause some issues because try/except might hide some errors that shouldn't be hidden
                        component._setup_geometry(parameterization_solver, ffd_geometric_variables, plot=plot, parent_translation=parent_translation)

                    except NotImplementedError:
                        warnings.warn(f"'_setup_geometry' has not been implemented for component {component._name} of {type(component)}")
            
            # If component has children, set up their geometries
            if component.comps:
                for comp_name, comp in component.comps.items():
                    setup_geometries(comp, parent_translation)

            return
    
        setup_geometries(self.system)

        for connection in self._geometric_connections:
            projection_1 = connection[0]
            projection_2 = connection[1]
            comp_1 : Component = connection[2]
            comp_2 : Component = connection[3]
            desired_value : csdl.Variable = connection[4]
            if isinstance(projection_1, list):
                connection = comp_1.geometry.evaluate(parametric_coordinates=projection_1) - comp_2.geometry.evaluate(parametric_coordinates=projection_2)
            elif isinstance(projection_1, np.ndarray):
                connection = comp_1._ffd_block.evaluate(parametric_coordinates=projection_1) - comp_2._ffd_block.evaluate(parametric_coordinates=projection_2)
            else:
                print(f"wrong type {type(projection_1)} for projection")
                raise NotImplementedError
            
            if desired_value is None:
                ffd_geometric_variables.add_variable(connection, connection.value)
                # print(connection.value)
            else:
                # if connection.shape != desired_value.shape:
                #     if desired_value.shape == (1, ):
                #         ffd_geometric_variables.add_variable(csdl.norm(connection), desired_value)
                #     else:
                #         raise ValueError(f"geometric connection has shape {connection.shape}, and desired value has shape {desired_value.shape}. If the shape of the deired value is (1, ), the norm of the connection will be enforced.")

                # else:
                    ffd_geometric_variables.add_variable(connection, desired_value)
        
        if additional_constraints:
            for constr in additional_constraints:
                connection = csdl.norm(self.system.geometry.evaluate(parametric_coordinates=constr[0]) - self.system.geometry.evaluate(parametric_coordinates=constr[1]))
                ffd_geometric_variables.add_variable(connection, constr[2])

        # Evalauate the parameterization solver
        t1 = time.time()
        if recorder is not None:
            print("Setting 'inline' to false for inner optimization")
            recorder.inline = False
        parameterization_solver.evaluate(ffd_geometric_variables)
        t2 = time.time()
        print("time for inner optimization", t2-t1)

        
        if plot:
            system_geometry.plot(show=True)
