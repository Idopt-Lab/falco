from flight_simulator.core.vehicle.components.rotor import Rotor as RotorComp
from flight_simulator.utils.import_geometry import import_geometry
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
from flight_simulator import REPO_ROOT_FOLDER, Q_

from unittest import TestCase
from pathlib import Path

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np


class TestRotorComp(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_rotor_geometry_init(self):
        
        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()

        TestRotor = RotorComp(
            radius=csdl.Variable(name="radius", shape=(1, ), value=5/2),
            skip_ffd=False, 
            parameterization_solver=parameterization_solver,
            ffd_geometric_variables=ffd_geometric_variables)


        self.assertEqual(TestRotor.parameters.radius.value, 5/2)

