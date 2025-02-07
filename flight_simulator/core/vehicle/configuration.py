from __future__ import annotations
from flight_simulator.core.vehicle.component import Component
from lsdo_function_spaces import FunctionSet
import numpy as np 
import csdl_alpha as csdl
import warnings
import copy
from typing import Union, List
import time


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

        # TODO: how to handle case if children components and parent components both have mass properties:
        #   Ex: if user assigns airframe mass and then also masses for airframe components
        system = self.system
        system_comps = system.comps

        # TODO: allow for parallel axis theorem
        if not np.array_equal(point, np.array([0. ,0., 0.])):
            raise NotImplementedError("Mass properties taken w.r.t. a specific point not yet implemented.")

        def _add_component_mps_to_system_mps(
                comps, 
                system_mass=0., 
                system_cg=np.zeros((3, )), 
                system_inertia_tensor=np.zeros((3, 3))
            ):
            """Add component-level mass properties to the system-level mass properties. 
            Only called internally by 'assemble_system_mass_properties'"""
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")

                    system_mass = system_mass * 1
                    system_cg = system_cg * 1
                    system_inertia_tensor = system_inertia_tensor * 1

                # otherwise add the mass properties
                else:
                    m = mass_props.mass
                    cg = mass_props.cg_vector
                    it = mass_props.inertia_tensor

                    # if isinstance(m, csdl.Variable):
                    #     print(f"{comp_name} mass", m.value)
                    # else:
                    #     print(f"{comp_name} mass", m)

                    if m is not None:
                        system_mass = system_mass + m
                    if cg is not None:
                        system_cg = (system_cg * system_mass + m * cg) / (system_mass +  m)
                    if it is not None:
                        system_inertia_tensor = system_inertia_tensor + it
                    elif m is not None and cg is not None:
                        x = cg[0]
                        y = cg[1]
                        z = cg[2]
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
                system_mass, system_cg, system_inertia_tensor = \
                _add_component_mps_to_system_mps(system_comps)

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
    
            system_mass, system_cg, system_inertia_tensor = \
                _add_component_mps_to_system_mps(system_comps, system_mass, system_cg, system_inertia_tensor)

            system.quantities.mass_properties.mass = system_mass
            system.quantities.mass_properties.cg_vector = system_cg
            system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

        # Update mass properties for any copied configurations
        if update_copies:
            # def _print_existing_mps(comp: Component):
            #     m = comp.quantities.mass_properties.mass
            #     cg_vector = comp.quantities.mass_properties.cg_vector
            #     inertia_tensor = comp.quantities.mass_properties.inertia_tensor

            #     if isinstance(m, csdl.Variable):
            #         print(m.value)
            #     else:
            #         print(m)
            #     if isinstance(cg_vector, csdl.Variable):
            #         print(cg_vector.value)
            #     else:
            #         print(cg_vector)

            #     if isinstance(inertia_tensor, csdl.Variable):
            #         print(inertia_tensor.value)
            #     else:
            #         print(inertia_tensor)

            #     print("\n")

            #     if comp.comps:
            #         for comp in comp.comps.values():
            #             _print_existing_mps(comp)

            # _print_existing_mps(system)

            def _update_comp_copy_mps(component_copy: Component, original_component: Component):
                # Get original mass properties
                original_mps = original_component.quantities.mass_properties
                
                # Copy original mass properties and set them as new ones
                component_copy.quantities.mass_properties = copy.copy(original_mps)

                # Repeat if component has children
                for child_name, child_copy in component_copy.comps.items():
                    if child_name in original_component.comps.keys():
                        original_child = original_component.comps[child_name]
                        _update_comp_copy_mps(child_copy, original_child)

            def _update_config_copy_mps(config_copy: self):
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
    ):
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

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2))
        
        # Else choose the center points of the FFD block
        else:
            point_1 = comp_1._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            point_2 = comp_2._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))

            projection_1 = comp_1.geometry.project(point_1)
            projection_2 = comp_2.geometry.project(point_2)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2))
        
        return

    def setup_geometry(self, run_ffd : bool=True, plot : bool=False):
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

        def setup_geometries(component: Component):
            # If component has a geometry, set up its geometry
            if component.geometry is not None:
                if component._skip_ffd is True:
                    pass
                else:
                    try: # NOTE: might cause some issues because try/except might hide some errors that shouldn't be hidden
                        component._setup_geometry(parameterization_solver, ffd_geometric_variables, plot=plot)

                    except NotImplementedError:
                        warnings.warn(f"'_setup_geometry' has not been implemented for component {component._name} of {type(component)}")
            
            # If component has children, set up their geometries
            if component.comps:
                for comp_name, comp in component.comps.items():
                    setup_geometries(comp)

            return
    
        setup_geometries(self.system)

        for connection in self._geometric_connections:
            projection_1 = connection[0]
            projection_2 = connection[1]
            comp_1 : Component = connection[2]
            comp_2 : Component = connection[3]
            if isinstance(projection_1, list):
                connection = comp_1.geometry.evaluate(parametric_coordinates=projection_1) - comp_2.geometry.evaluate(parametric_coordinates=projection_2)
            elif isinstance(projection_1, np.ndarray):
                connection = comp_1._ffd_block.evaluate(parametric_coordinates=projection_1) - comp_2._ffd_block.evaluate(parametric_coordinates=projection_2)
            else:
                print(f"wrong type {type(projection_1)} for projection")
                raise NotImplementedError
            
            ffd_geometric_variables.add_variable(connection, connection.value)

        # Evalauate the parameterization solver
        t1 = time.time()
        parameterization_solver.evaluate(ffd_geometric_variables)
        t2 = time.time()
        print("time for inner optimization", t2-t1)
        
        if plot:
            system_geometry.plot(show=True)

        