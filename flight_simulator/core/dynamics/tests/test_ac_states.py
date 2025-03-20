from unittest import TestCase
from flight_simulator import ureg
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.aircraft_states import AircraftStates


class AxisTests(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        self.ac_states = AircraftStates()

    def test_create_ac_states_object(self):
        self.ac_states.phi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='phi')
        self.ac_states.theta = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(4.), ]), name='theta')
        self.ac_states.psi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='psi')

        fd_axis = Axis(
            name='Flight Dynamics Body Fixed Axis',
            translation=np.array([0, 0, 5000]) * ureg.ft,
            phi=self.ac_states.phi,
            theta=self.ac_states.theta,
            psi=self.ac_states.psi,
            sequence=np.array([3, 2, 1]),
            origin=ValidOrigins.Inertial.value,
        )
        np.testing.assert_almost_equal(np.rad2deg(fd_axis.euler_angles.value), desired=np.array([0., 4., 0.]))

    def test_update_ac_states_value(self):
        self.ac_states.phi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='phi')
        self.ac_states.theta = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(4.), ]), name='theta')
        self.ac_states.psi = csdl.Variable(shape=(1,), value=np.array([np.deg2rad(0.), ]), name='psi')

        fd_axis = Axis(
            name='Flight Dynamics Body Fixed Axis',
            translation=np.array([0, 0, 5000]) * ureg.ft,
            phi=self.ac_states.phi,
            theta=self.ac_states.theta,
            psi=self.ac_states.psi,
            sequence=np.array([3, 2, 1]),
            origin=ValidOrigins.Inertial.value,
        )
        np.testing.assert_almost_equal(np.rad2deg(fd_axis.euler_angles.value), desired=np.array([0., 4., 0.]))

        self.ac_states.phi.set_value(value=np.deg2rad(2.))
        np.testing.assert_almost_equal(np.rad2deg(fd_axis.euler_angles.value), desired=np.array([2., 4., 0.]))


