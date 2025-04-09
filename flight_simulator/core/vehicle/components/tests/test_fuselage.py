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

        TestFuselage = FuseComp(
            length=csdl.Variable(name="length", shape=(1, ), value=8),
            max_height=csdl.Variable(name="max_height", shape=(1, ), value=1),
            max_width=csdl.Variable(name="max_width", shape=(1, ), value=1.24),
            geometry=fuselage, skip_ffd=False, 
            parameterization_solver=parameterization_solver,
            ffd_geometric_variables=ffd_geometric_variables)

        self.assertIsNotNone(TestFuselage.geometry)

        self.assertEqual(TestFuselage.parameters.length.value, 8)
        self.assertEqual(TestFuselage.parameters.max_height.value, 1)
        self.assertEqual(TestFuselage.parameters.max_width.value, 1.24)
        
        parameterization_solver.evaluate(ffd_geometric_variables)
        geometry.plot()
