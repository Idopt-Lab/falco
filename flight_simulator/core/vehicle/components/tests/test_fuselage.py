from flight_simulator.core.vehicle.components.fuselage import Fuselage as FuseComp
from flight_simulator.utils.import_geometry import import_geometry
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
from flight_simulator import REPO_ROOT_FOLDER, Q_

from unittest import TestCase
from pathlib import Path

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np


class TestFuseComp(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_fuse_geometry_init(self):
        
        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()

        ft2m = 0.3048

        geometry = import_geometry(
            file_name='example_fuselage.stp',
            file_path=REPO_ROOT_FOLDER / 'flight_simulator' / 'core' / 'vehicle' / 'components' / 'tests' / 'test_geometries',
            refit=False,
            scale=ft2m,
            rotate_to_body_fixed_frame=True
        )
        fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')

        fuselage_nose_guess = np.array([-1.75, 0, -4])*ft2m
        fuselage_rear_guess = np.array([-29.5, 0, -5.5])*ft2m
        fuselage_nose_pts_parametric = fuselage.project(fuselage_nose_guess, grid_search_density_parameter=20, plot=False)
        fuselage_nose = geometry.evaluate(fuselage_nose_pts_parametric)
        fuselage_rear_pts_parametric = fuselage.project(fuselage_rear_guess, plot=False)
        fuselage_rear = geometry.evaluate(fuselage_rear_pts_parametric)


        TestFuselage = FuseComp(
            length=csdl.Variable(name="length", shape=(1, ), value=8.2242552),
            max_height=csdl.Variable(name="max_height", shape=(1, ), value=1.09),
            max_width=csdl.Variable(name="max_width", shape=(1, ), value=1.24070602),
            geometry=fuselage, skip_ffd=False, 
            parameterization_solver=parameterization_solver,
            ffd_geometric_variables=ffd_geometric_variables)

        self.assertIsNotNone(TestFuselage.geometry)

        self.assertEqual(TestFuselage.parameters.length.value, 8.2242552)
        self.assertEqual(TestFuselage.parameters.max_height.value, 1.09)
        self.assertEqual(TestFuselage.parameters.max_width.value, 1.24070602)
