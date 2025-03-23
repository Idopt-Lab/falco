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
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from dataclasses import dataclass
import time
import warnings
import copy
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.vehicle.models.propulsion.propulsion_model import HLPropCurve, CruisePropCurve, AircraftPropulsion
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel, AircraftAerodynamics
from flight_simulator.core.vehicle.models.mass_properties.mass_prop_model import GravityLoads

class ComponentQuantities:
    def __init__(self, mass_properties: MassProperties = None) -> None:
        self._mass_properties = mass_properties
        self.surface_mesh = []
        self.surface_area = None
        self.aero_forces = None
        self.aero_moments = None
        self.prop_forces = None
        self.prop_moments = None
        self.grav_forces = None
        self.grav_moments = None
        self.total_forces = None
        self.total_moments = None
        self.lift_model = None
        self.prop_radius = None
        self.prop_curve = None
        self.prop_axis = None
        self.wing_axis = None

        if mass_properties is None:
            default_axis = Axis(name="Default Axis", origin=ValidOrigins.Inertial.value)
            default_inertia = MassMI(axis=default_axis)
            default_cg = Vector(vector=Q_(np.zeros(3), 'm'), axis=default_axis)
            self.mass_properties = MassProperties(cg=default_cg, inertia=default_inertia)

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

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot: bool = False):
        if not hasattr(self, 'geometry') or self.geometry is None:
            return
        rigid_body_translation = csdl.ImplicitVariable(shape=(3,), value=0., name=f'{self._name}_rigid_body_translation')

        for function in self.geometry.functions.values():
            if function.name == "rigid_body_translation":
                shape = function.coefficients.shape
                function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action="k->ijk")


        parameterization_solver.add_parameter(rigid_body_translation, cost=0.001)


    def compute_aero_loads(self, fd_state, controls, fd_axis):
        """
        Compute aerodynamic loads (forces and moments) for this component 
        if an aerodynamic model is attached.
        Returns:
            forces (np.array): Aerodynamic forces vector.
            moments (np.array): Aerodynamic moments vector.
        """
        aero = AircraftAerodynamics(
            fd_state=fd_state,
            wing_axis=self.quantities.wing_axis,  
            controls=controls,
            lift_model=self.quantities.lift_model
        )

        # Compute aerodynamic loads using the aero model
        fm = aero.get_FM_refPoint(x_bar=fd_state, u_bar=controls)
        fm_rotated = fm.rotate_to_axis(fd_axis)
        self.quantities.aero_forces = fm_rotated.F.vector.value
        self.quantities.aero_moments = fm_rotated.M.vector.value
        # print(fm_rotated.F.vector.value, fm_rotated.M.vector.value)
        return fm_rotated.F.vector.value, fm_rotated.M.vector.value
    

    def compute_propulsion_loads(self, fd_state, controls, fd_axis):
        """
        Compute propulsion loads (forces and moments) for this component if it 
        has propulsion capabilities.
        Returns:
            forces (np.array): Propulsion forces vector.
            moments (np.array): Propulsion moments vector.
        """
       
        motor_prop = AircraftPropulsion(
            prop_axis=self.quantities.prop_axis,
            fd_state=fd_state,
            controls=controls,
            radius=self.quantities.prop_radius,
            prop_curve=self.quantities.prop_curve
        )
        fm = motor_prop.get_FM_refPoint(x_bar=fd_state, u_bar=controls)
        fm_rotated = fm.rotate_to_axis(fd_axis)
        self.quantities.prop_forces = fm_rotated.F.vector.value
        self.quantities.prop_moments = fm_rotated.M.vector.value
        # print(fm_rotated.F.vector.value, fm_rotated.M.vector.value)
        return fm_rotated.F.vector.value, fm_rotated.M.vector.value

    def compute_gravity_loads(self, fd_axis, fd_state, controls):
        """
        Compute gravity loads (forces and moments) for this component.
        Returns:
            forces (np.array): Gravity forces vector.
            moments (np.array): Gravity moments vector.
        """
        gravity_load = GravityLoads(fd_state=fd_state, fd_axis=fd_axis, controls=controls, component=self)
        fm = gravity_load.get_FM_refPoint(x_bar=fd_state, u_bar=controls)
        self.quantities.grav_forces = fm.F.vector.value
        self.quantities.grav_moments = fm.M.vector.value
        # print(fm.F.vector.value, fm.M.vector.value)
        return fm.F.vector.value, fm.M.vector.value
    
    def compute_total_loads(self, fd_state, controls, fd_axis):
        """
        Compute the total loads (forces and moments) for this component by summing the
        aerodynamic, propulsion, and gravity loads.
        Returns:
            total_forces (np.array): Sum of forces from all loads.
            total_moments (np.array): Sum of moments from all loads.
        """
        # Get aerodynamic loads only if a lift model is provided and the component uses one
        if self.quantities.lift_model is not None:
            aero_forces, aero_moments = self.compute_aero_loads(fd_state, controls, fd_axis)
        else:
            aero_forces = np.zeros(3)
            aero_moments = np.zeros(3)
            
        # Get propulsion loads only if radius and prop_curve are provided and the component uses propulsion
        if self.quantities.prop_curve is not None:
            prop_forces, prop_moments = self.compute_propulsion_loads(fd_state, controls, fd_axis)
            prop_forces = np.nan_to_num(prop_forces, nan=0.0)
            prop_moments = np.nan_to_num(prop_moments, nan=0.0)
        else:
            prop_forces = np.zeros(3)
            prop_moments = np.zeros(3)

        # Get loads from gravity model
        grav_forces, grav_moments = self.compute_gravity_loads(fd_axis, fd_state, controls)
        
        # Sum the forces and moments
        total_forces = aero_forces + prop_forces + grav_forces
        total_moments = aero_moments + prop_moments + grav_moments

        # print(total_forces, total_moments)

        self.quantities.total_forces = total_forces
        self.quantities.total_moments = total_moments
        
        return total_forces, total_moments
    
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


    