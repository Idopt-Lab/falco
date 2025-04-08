from flight_simulator.core.vehicle.components.wing import Wing
from flight_simulator.utils.import_geometry import import_geometry
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
from flight_simulator import REPO_ROOT_FOLDER, Q_

from unittest import TestCase
from pathlib import Path

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np


class TestWingComp(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_wing_geometry_init(self):
        
        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()

        ft2m = 0.3048

        wing_geometry = import_geometry(
            file_name='example_wing.stp',
            file_path=REPO_ROOT_FOLDER / 'flight_simulator' / 'core' / 'vehicle' / 'components' / 'tests' / 'test_geometries',
            refit=False,
            scale=ft2m,
            rotate_to_body_fixed_frame=True
        )
        wing = wing_geometry.declare_component(function_search_names=['WingGeom'], name='wing')


        # Wing Region Info
        wing_le_left_guess = np.array([0, -10, 0])*ft2m
        wing_le_left_parametric = wing.project(wing_le_left_guess, plot=False)

        wing_le_right_guess = np.array([0, 10, 0])*ft2m
        wing_le_right_parametric = wing.project(wing_le_right_guess, plot=False)

        wing_le_center_guess = np.array([0, 0., 0])*ft2m
        wing_le_center_parametric = wing.project(wing_le_center_guess, plot=False)

        wing_te_left_guess = np.array([-1, -10, 0])*ft2m
        wing_te_left_parametric = wing.project(wing_te_left_guess, plot=False)

        wing_te_right_guess = np.array([-1, 10, 0])*ft2m
        wing_te_right_parametric = wing.project(wing_te_right_guess, plot=False)

        wing_te_center_guess = np.array([-1, 0., 0])*ft2m
        wing_te_center_parametric = wing.project(wing_te_center_guess, plot=False)

        wing_qc_center_parametric = wing_geometry.project(np.array([-0.25, 0., 0])*ft2m, plot=False)
        wing_qc_tip_left_parametric = wing_geometry.project(np.array([-0.25, -10., 0])*ft2m, plot=False)
        wing_qc_tip_right_parametric = wing_geometry.project(np.array([-0.25, 0., 0])*ft2m, plot=False)

        wing_parametric_geometry = [
            wing_le_left_parametric,
            wing_le_right_parametric,
            wing_le_center_parametric,
            wing_te_left_parametric,
            wing_te_right_parametric,
            wing_te_center_parametric,
            wing_qc_center_parametric,
            wing_qc_tip_right_parametric,
            wing_qc_tip_left_parametric
        ]

        test_wing_area = csdl.Variable(name="wing_area", shape=(1, ), value=10.0)
        test_wing_span = csdl.Variable(name="wing_span", shape=(1, ), value=10.0)
        test_wing_sweep = csdl.Variable(name="wing_sweep", shape=(1, ), value=0)
        test_wing_dihedral = csdl.Variable(name="wing_dihedral", shape=(1, ), value=0)
        TestWing = Wing(
            S_ref = test_wing_area,
            span = test_wing_span,
            sweep = test_wing_sweep,
            dihedral = test_wing_dihedral,
            geometry = wing_geometry,
            parameterization_solver=parameterization_solver,
            ffd_geometric_variables=ffd_geometric_variables,
            parametric_geometry=wing_parametric_geometry,
            name='test wing',
            orientation='horizontal',
            tight_fit_ffd=False
        )
        self.assertIsNotNone(TestWing.geometry)

        self.assertEqual(TestWing.parameters.S_ref.value, 10.0)
        self.assertEqual(TestWing.parameters.sweep.value, 0)
        self.assertEqual(TestWing.parameters.dihedral.value, 0)
        self.assertEqual(TestWing.parameters.span.value, 10.0)
