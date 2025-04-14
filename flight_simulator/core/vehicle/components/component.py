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
from flight_simulator.core.vehicle.models.propulsion.propulsion_model import HLPropCurve, CruisePropCurve, AircraftPropulsion
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel, AircraftAerodynamics
from flight_simulator.core.loads.loads import Loads


class ComponentQuantities:
    def __init__(self, mass_properties: MassProperties = None) -> None:
        self._mass_properties = mass_properties
        self.surface_mesh = []
        self.surface_area = None
        self.load_solvers = []

        if mass_properties is None:
            self.mass_properties = MassProperties.create_default_mass_properties()


    def __repr__(self):

        output = (
            f"Component Mass: {self.mass_properties.mass.value} kg\n"
            f"Component CG: {self.mass_properties.cg_vector.vector.value} m\n"
            f"Component Inertia: {self.mass_properties.inertia_tensor.inertia_tensor.value}"
        )
        return output


@dataclass
class ComponentParameters:
    actuate_angle: Union[csdl.Variable, float, None] = None
    pass    


class Component:
    def __init__(self, name: str, geometry: Union[FunctionSet, None] = None, 
                compute_surface_area_flag: bool = False, 
                parameterization_solver: ParameterizationSolver=None, 
                ffd_geometric_variables: GeometricVariables=None, **kwargs) -> None:
        self._name = name
        self.parent = None
        self.comps = {}
        self.geometry: Union[FunctionSet, Geometry, None] = geometry
        self.compute_surface_area_flag = compute_surface_area_flag
        self.quantities: ComponentQuantities = ComponentQuantities()
        self.parameters: ComponentParameters = ComponentParameters()
        self._parameterization_solver = parameterization_solver
        self._ffd_geometric_variables = ffd_geometric_variables 

        for key, value in kwargs.items():
            setattr(self.parameters, key, value)

        if geometry and compute_surface_area_flag:
            self.quantities.surface_area = self._compute_surface_area(geometry)

        if geometry is not None and isinstance(geometry, FunctionSet):
            if "do_not_remake_ffd_block" not in kwargs:
                num_ffd_sections = 3
                self._ffd_block = construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))
    


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

        total_forces = csdl.Variable(shape=(3,), value=0.)
        total_moments = csdl.Variable(shape=(3,), value=0.)


        if bool(self.quantities.load_solvers):
            for ls in self.quantities.load_solvers:
                assert isinstance(ls,Loads)
                fm = ls.get_FM_refPoint(x_bar=fd_state, u_bar=controls)
                fm_rotated = fm.rotate_to_axis(fd_state.axis)
                total_forces += fm_rotated.F.vector
                total_moments += fm_rotated.M.vector
        else:
            pass

        for comp in self.comps.values():
            f, m = comp.compute_total_loads(fd_state, controls)
            total_forces += f 
            total_moments += m

        grav_loads = GravityLoads(fd_state=fd_state, controls=controls, mass_properties=self.quantities.mass_properties)
        gfm = grav_loads.get_FM_refPoint(x_bar=fd_state, u_bar=controls)
        print(f"Gravity Loads: {gfm.F.vector.value} N")
        total_forces += gfm.F.vector
        total_moments += gfm.M.vector

        return total_forces, total_moments
    
    def compute_inertia(self):
        """
        Compute the inertia of this component.
        Returns:
            inertia (np.array): Inertia tensor of the component.
        """
        cg = self.quantities.mass_properties.cg_vector.vector
        axis = self.quantities.mass_properties.cg_vector.axis

        if hasattr(self.quantities.mass_properties.mass, 'value'):
            mass = self.quantities.mass_properties.mass.value
        else:
            mass = self.quantities.mass_properties.mass.magnitude

        x= cg[0].value
        y= cg[1].value
        z= cg[2].value


        inertia = np.zeros((3, 3))

        Ixx = inertia[0,0] + mass * (y**2 + z**2)
        Iyy = inertia[1,1] + mass * (x**2 + z**2)
        Izz = inertia[2,2] + mass * (x**2 + y**2)
        Ixy = inertia[0,1] - mass * x * y
        Ixz = inertia[0,2] - mass * x * z
        Iyz = inertia[1,2] - mass * y * z
        Iyx = Ixy
        Izx = Ixz
        Izy = Iyz

        inertia = csdl.Variable(shape=(3, 3), value=0.)
        inertia = inertia.set(csdl.slice[0, 0], Ixx)
        inertia = inertia.set(csdl.slice[0, 1], Ixy)
        inertia = inertia.set(csdl.slice[0, 2], Ixz)
        inertia = inertia.set(csdl.slice[1, 0], Iyx)
        inertia = inertia.set(csdl.slice[1, 1], Iyy)
        inertia = inertia.set(csdl.slice[1, 2], Iyz)
        inertia = inertia.set(csdl.slice[2, 0], Izx)
        inertia = inertia.set(csdl.slice[2, 1], Izy)
        inertia = inertia.set(csdl.slice[2, 2], Izz)

        MI = MassMI(axis=axis,
            Ixx=Q_(inertia[0,0].value, 'kg*(m*m)'),
            Ixy=Q_(inertia[0,1].value, 'kg*(m*m)'),
            Ixz=Q_(inertia[0,2].value, 'kg*(m*m)'),
            Iyy=Q_(inertia[1,1].value, 'kg*(m*m)'),
            Iyz=Q_(inertia[1,2].value, 'kg*(m*m)'),
            Izz=Q_(inertia[2,2].value, 'kg*(m*m)'),
            )

        self.quantities.mass_properties.inertia_tensor = MI
        
        return MI
    
    def compute_mass_properties(self):

        """
        Compute the mass properties of this component.
        Returns:
            mass_properties (dict): Dictionary containing the mass, center of gravity, and inertia tensor.
        """
    
        mass = self.quantities.mass_properties.mass
        cg = self.quantities.mass_properties.cg_vector
        inertia = self.compute_inertia()
    

        total_mass = mass.magnitude
        total_cg = cg.vector * mass.magnitude
        total_inertia = inertia.inertia_tensor


        for comp in self.comps.values():
            comp_mass_properties = comp.compute_mass_properties()
            comp_mass = comp_mass_properties.mass
            comp_cg = comp_mass_properties.cg_vector
            comp_inertia = comp_mass_properties.inertia_tensor
            total_mass += comp_mass.value
            total_cg += comp_cg.vector * comp_mass.value
            total_inertia += comp_inertia.inertia_tensor


        overall_cg_vector = total_cg / total_mass
        overall_cg = type(cg)(vector=overall_cg_vector, axis=cg.axis)
    

        overall_inertia = type(inertia)(
            axis=cg.axis,
            Ixx=total_inertia[0, 0],
            Ixy=-total_inertia[0, 1],
            Ixz=-total_inertia[0, 2],
            Iyy=total_inertia[1, 1],
            Iyz=-total_inertia[1, 2],
            Izz=total_inertia[2, 2])
        total_mass_properties = MassProperties(mass=Q_(total_mass,'kg'),cg=overall_cg,inertia=overall_inertia)

        return total_mass_properties

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


    