from unittest import TestCase
from falco import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.dynamics.aircraft_states import AircraftStates


class AxisTests(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.inertial_axis = Axis(
            name='Inertial Axis',
            origin=ValidOrigins.Inertial.value
        )

    def test_create_ac_states_object(self):
        phi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='phi')
        theta = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(4.), ]), name='theta')
        psi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='psi')

        fd_axis = Axis(
            name='Flight Dynamics Body Fixed Axis',
            x=Q_(0, 'ft'),
            y=Q_(0, 'ft'),
            z=Q_(5000, 'ft'),
            phi=phi,
            theta=theta,
            psi=psi,
            sequence=np.array([3, 2, 1]),
            reference=self.inertial_axis,
            origin=ValidOrigins.Inertial.value,
        )

        ac_states = AircraftStates(axis=fd_axis)
        np.testing.assert_almost_equal(np.rad2deg(ac_states.euler_angles_vector.value), desired=np.array([0., 4., 0.]))
        np.testing.assert_almost_equal(ac_states.position_vector.value, desired=np.array([0., 0., 1524.]))

    def test_update_ac_states_value(self):
        phi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='phi')
        theta = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(4.), ]), name='theta')
        psi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='psi')

        fd_axis = Axis(
            name='Flight Dynamics Body Fixed Axis',
            x=Q_(0, 'ft'),
            y=Q_(0, 'ft'),
            z=Q_(5000, 'ft'),
            phi=phi,
            theta=theta,
            psi=psi,
            sequence=np.array([3, 2, 1]),
            reference=self.inertial_axis,
            origin=ValidOrigins.Inertial.value,
        )

        ac_states = AircraftStates(axis=fd_axis)
        ac_states.euler_angles_vector.set_value(value=np.deg2rad([2., 4., 0.]))
        np.testing.assert_almost_equal(np.rad2deg(fd_axis.euler_angles_vector.value), desired=np.array([2., 4., 0.]))


