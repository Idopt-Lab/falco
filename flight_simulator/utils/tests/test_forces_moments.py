from unittest import TestCase

from sympy.physics.mechanics.functions import inertia

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.utils.axis import Axis
from flight_simulator.utils.forces_moments import Vector, ForcesMoments


class ForcesMomentsTests(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            angles=np.array([0, 0, 0]) * ureg.degree,
            origin='inertial'
        )

        self.fd_axis = Axis(
            name='Flight Dynamics Body Fixed Axis',
            translation=np.array([0, 0, 5000]) * ureg.ft,
            angles=Q_(np.asarray([0, 0, 0]), 'deg'),
            sequence=np.array([3, 2, 1]),
            reference=self.inertial_axis,
            origin='cg'
        )

        self.axis1 = Axis(
            name='Axis 1',
            translation=np.array([5, 0, 0]) * ureg.meter,
            angles=np.array([0, 0, 0]) * ureg.degree,
            sequence=np.array([3, 2, 1]),
            reference=self.fd_axis,
            origin='ref'
        )

        self.axis2 = Axis(
            name='Axis 1',
            translation=np.array([0, 5, -2]) * ureg.meter,
            angles=np.array([0, 0, 0]) * ureg.degree,
            sequence=np.array([3, 2, 1]),
            reference=self.fd_axis,
            origin='ref'
        )

    def test_forces_moments_obj(self):
        force_vector_1 = Vector(vector=np.array([100, 400, 0]) * ureg.lbf, axis=self.inertial_axis)
        np.testing.assert_almost_equal(force_vector_1.magnitude.value, desired=1834.0481, decimal=3)

    def test_ForcesMoments_obj(self):
        force_vector_1 = Vector(vector=np.array([0, 0, 0]) * ureg.newton, axis=self.axis1)
        moment_vector_1 = Vector(vector=np.array([0, 0, 0]) * ureg.newton*ureg.meter, axis=self.axis1)
        load1 = ForcesMoments(force=force_vector_1, moment=moment_vector_1)

    def test_loads_transfer_1(self):
        force_vector_1 = Vector(vector=np.array([0, 0, 10]) * ureg.newton, axis=self.axis1)
        moment_vector_1 = Vector(vector=np.array([0, 0, 0]) * ureg.newton*ureg.meter, axis=self.axis1)
        load1 = ForcesMoments(force=force_vector_1, moment=moment_vector_1)

        new_load = load1.rotate_to_axis(parent_or_child_axis=self.fd_axis)
        print(new_load.F.vector.value)
        print(new_load.M.vector.value)