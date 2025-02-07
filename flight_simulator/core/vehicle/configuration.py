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
            point_1 = comp_1.ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            point_2 = comp_2.ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))

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
                connection = comp_1.ffd_block.evaluate(parametric_coordinates=projection_1) - comp_2.ffd_block.evaluate(parametric_coordinates=projection_2)
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

        