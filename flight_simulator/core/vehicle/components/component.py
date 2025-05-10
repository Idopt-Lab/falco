from __future__ import annotations
from pathlib import Path
from flight_simulator.core.loads.mass_properties import MassProperties
from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from flight_simulator import ureg, Q_
from typing import Union, List
import numpy as np
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
import csdl_alpha as csdl
from dataclasses import dataclass
import time
import warnings
import copy
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI, GravityLoads
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.loads.loads import Loads
from flight_simulator.utils.euler_rotations import build_rotation_matrix



@dataclass
class ComponentParameters:
    actuate_angle: Union[csdl.Variable, float, None] = None
    pass    


class Component:
    def __init__(self, name: str, geometry: Union[FunctionSet, None] = None, 
                compute_surface_area_flag: bool = False, 
                parameterization_solver: ParameterizationSolver=None,
                mass_properties: MassProperties = None, 
                ffd_geometric_variables: GeometricVariables=None, **kwargs) -> None:
        self._name = name
        self.parent = None
        self.comps = {}
        self.geometry: Union[FunctionSet, Geometry, None] = geometry
        self.compute_surface_area_flag = compute_surface_area_flag
        self.parameters: ComponentParameters = ComponentParameters()
        self._parameterization_solver = parameterization_solver
        self._ffd_geometric_variables = ffd_geometric_variables 
        self.mass_properties = mass_properties
        self.surface_mesh = []
        self.surface_area = None
        self.load_solvers = []

        for key, value in kwargs.items():
            setattr(self.parameters, key, value)

        if geometry and compute_surface_area_flag:
            self.surface_area = self._compute_surface_area(geometry)

        if geometry is not None and isinstance(geometry, FunctionSet):
            if "do_not_remake_ffd_block" not in kwargs:
                num_ffd_sections = 3
                self._ffd_block = construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))

        # if mass_properties is None:
        #     self.mass_properties = MassProperties.create_default_mass_properties()
    

    def __repr__(self):

        output = (
            f"Component Mass: {self.mass_properties.mass.value} kg\n"
            f"Component CG: {self.mass_properties.cg_vector.vector.value} m\n"
            f"Component Inertia: {self.mass_properties.inertia_tensor.inertia_tensor.value}"
        )
        return output


    def add_subcomponent(self, subcomponent: Component):
        if not isinstance(subcomponent, Component):
            raise TypeError(f"Subcomponent must be of type 'Component'. Received: {type(subcomponent)}")

        if subcomponent._name in self.comps:
            raise KeyError(f"Subcomponent with name '{subcomponent._name}' already exists.")

        if subcomponent.parent is not None:
            raise ValueError(f"Subcomponent '{subcomponent._name}' is already a subcomponent of another component.")

        self.comps[subcomponent._name] = subcomponent
        subcomponent.parent = self

    def plot(self, show: bool = False, **kwargs):
        """Plot a component's geometry."""
        if self.geometry is None:
            raise ValueError(f"Cannot plot component {self} since its geometry is None.")
        else:
            self.geometry.plot(show=show, **kwargs)

    def remove_subcomponent(self, subcomponent: Component):
        for name, comp in self.comps.items():
            if comp == subcomponent:
                self.comps.pop(name)
                subcomponent.parent = None
                return
        raise KeyError(f"Subcomponent '{subcomponent._name}' not found.")

    def visualize_component_hierarchy(self, file_name: str = "component_hierarchy", file_format: str = "png", filepath: Path = Path.cwd(), show: bool = False):
        csdl.check_parameter(file_name, "file_name", types=str)
        csdl.check_parameter(file_format, "file_format", values=("png", "pdf"))
        csdl.check_parameter(show, "show", types=bool)
        try:
            from graphviz import Graph
        except ImportError:
            raise ImportError("Must install graphviz via application and do 'pip install graphviz'")

        graph = Graph(comment="Component Hierarchy")

        def add_component_to_graph(comp: Component, comp_name: str, parent_name=None):
            graph.node(comp_name, comp_name)
            if parent_name is not None:
                graph.edge(parent_name, comp_name)
            for child_name, child in comp.comps.items():
                add_component_to_graph(child, child_name, comp_name)

        add_component_to_graph(self, self._name)
        graph.render(file_name, directory=filepath, format=file_format, view=show)

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot: bool = False):
        if not hasattr(self, 'geometry') or self.geometry is None:
            return
        rigid_body_translation = csdl.ImplicitVariable(shape=(3,), value=0., name=f'{self._name}_rigid_body_translation')

        for function in self.geometry.functions.values():
            if function.name == "rigid_body_translation":
                shape = function.coefficients.shape
                function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action="k->ijk")


        parameterization_solver.add_parameter(rigid_body_translation, cost=0.001)

    
    def compute_total_loads(self, fd_state, controls):
        """
        Compute the total loads (forces and moments) for this component by summing the
        aerodynamic, propulsion, and gravity loads.
        Returns:
            total_forces (np.array): Sum of forces from all loads.
            total_moments (np.array): Sum of moments from all loads.
        """
        from flight_simulator.utils.euler_rotations import build_rotation_matrix

        total_forces = csdl.Variable(shape=(3,), value=0.)
        total_moments = csdl.Variable(shape=(3,), value=0.)


        if bool(self.load_solvers):
            for ls in self.load_solvers:
                assert isinstance(ls, Loads)
                local_axis = self.mass_properties.cg_vector.axis
                fm = ls.get_FM_localAxis(states=fd_state, controls=controls, axis=local_axis)

                if fm.F.axis.reference.name == 'Inertial Axis' or fm.F.axis.reference.name == 'OpenVSP Axis':
                    # Let's rotate the forces and moments to the inertial axis
                    fm_fd_axis = fm.transform_to_axis(local_axis.reference, translate_flag=True, rotate_flag=True)
                elif fm.F.axis.name == 'Wind Axis':
                    # Implement R cross F
                    fm_fd_axis_component_origin = fm.transform_to_axis(fd_state.axis, translate_flag=False,
                                                                    rotate_flag=True)

                    F_trans, M_trans = ForcesMoments.translate_to_axis(fm_fd_axis_component_origin.F.vector, fm_fd_axis_component_origin.M.vector, local_axis.translation)

                    fm_fd_axis = ForcesMoments(force=Vector(vector=F_trans, axis=fm.F.axis),
                                             moment=Vector(vector=M_trans, axis=fm.F.axis))
                else:
                    raise NotImplementedError

                total_forces += fm_fd_axis.F.vector
                total_moments += fm_fd_axis.M.vector

        else:
            pass

        for comp in self.comps.values():
            f, m = comp.compute_total_loads(fd_state=fd_state, controls=controls)
            total_forces += f 
            total_moments += m

        if self.parent is None:
            if self.mass_properties is not None:
                grav_loads = GravityLoads(fd_state=fd_state, controls=controls, mass_properties=self.mass_properties)
                gfm = grav_loads.get_FM_localAxis()
                total_forces += gfm.F.vector
                total_moments += gfm.M.vector

        return total_forces, total_moments
    


    def compute_total_mass_properties(self):
    
        """
        Compute the total mass properties of this component, including its local inertia.
        This method computes the inertia using the component's mass and center of gravity,
        then recursively sums the properties from all subcomponents.

        Returns:
            MassProperties: Object containing the total mass, overall center of gravity,
                            and overall inertia tensor.
        """
        
        cg = self.mass_properties.cg_vector.vector
        axis = self.mass_properties.cg_vector.axis
        mass = self.mass_properties.mass

        x = cg[0]
        y = cg[1]
        z = cg[2]

        Ixx = mass * (y**2 + z**2)
        Iyy = mass * (x**2 + z**2)
        Izz = mass * (x**2 + y**2)
        Ixy = -mass * x * y
        Ixz = -mass * x * z
        Iyz = -mass * y * z

        inertia = csdl.Variable(shape=(3, 3), value=0.)
        inertia = inertia.set(csdl.slice[0, 0], Ixx)
        inertia = inertia.set(csdl.slice[0, 1], Ixy)
        inertia = inertia.set(csdl.slice[0, 2], Ixz)
        inertia = inertia.set(csdl.slice[1, 0], Ixy)
        inertia = inertia.set(csdl.slice[1, 1], Iyy)
        inertia = inertia.set(csdl.slice[1, 2], Iyz)
        inertia = inertia.set(csdl.slice[2, 0], Ixz)
        inertia = inertia.set(csdl.slice[2, 1], Iyz)
        inertia = inertia.set(csdl.slice[2, 2], Izz)

        local_MI = MassMI(
            axis=axis,
            Ixx=inertia[0, 0],
            Ixy=inertia[0, 1],
            Ixz=inertia[0, 2],
            Iyy=inertia[1, 1],
            Iyz=inertia[1, 2],
            Izz=inertia[2, 2],
        )
        self.mass_properties.inertia_tensor = local_MI

        # Initialize total properties with the local properties
        total_mass = mass
        total_cg = self.mass_properties.cg_vector.vector * mass
        total_inertia = local_MI.inertia_tensor

        # Recursively add mass properties from all subcomponents
        for comp in self.comps.values():
            comp_props = comp.compute_total_mass_properties()
            comp_mass = comp_props.mass
            comp_cg = comp_props.cg_vector
            comp_inertia = comp_props.inertia_tensor
            total_mass += comp_mass
            total_cg += comp_cg.vector * comp_mass
            total_inertia += comp_inertia.inertia_tensor

        overall_cg_vector = total_cg / total_mass
        overall_cg = type(self.mass_properties.cg_vector)(vector=overall_cg_vector, axis=axis)

        overall_inertia = type(local_MI)(
            axis=axis,
            Ixx=total_inertia[0, 0],
            Ixy=-total_inertia[0, 1],
            Ixz=-total_inertia[0, 2],
            Iyy=total_inertia[1, 1],
            Iyz=-total_inertia[1, 2],
            Izz=total_inertia[2, 2]
        )
        total_mass_properties = MassProperties(
            mass=Q_(total_mass.value, 'kg'),
            cg=overall_cg,
            inertia=overall_inertia
        )
    
        return total_mass_properties

    def _compute_surface_area(self, geometry: Geometry, plot_flag: bool = False):
        parametric_mesh_grid_num = 10
        surfaces = geometry.functions
        surface_area = csdl.Variable(shape=(1,), value=1)
        surface_mesh = self.surface_mesh
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


    