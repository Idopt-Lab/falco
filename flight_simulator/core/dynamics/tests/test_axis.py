from unittest import TestCase
from flight_simulator import ureg
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins


class AxisTests(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_create_axis(self):
        inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            origin=ValidOrigins.Inertial.value
        )
        np.testing.assert_almost_equal(inertial_axis.translation.value, desired=np.zeros(3,))

    def test_set_axis_translation_via_pint(self):
        inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            origin=ValidOrigins.Inertial.value
        )
        inertial_axis.translation = np.array([3, 0, 0]) * ureg.ft
        np.testing.assert_almost_equal(inertial_axis.translation.value, desired=np.array([3*0.3048, 0, 0]), decimal=5)

    def test_set_axis_translation_via_csdl(self):
        inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            origin=ValidOrigins.Inertial.value
        )
        inertial_axis.translation.set_value(np.array([0, 5, 0]))
        np.testing.assert_almost_equal(inertial_axis.translation.value, desired=np.array([0, 5, 0]))

    def test_create_axis_with_specified_euler_angles(self):
        axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            origin=ValidOrigins.Inertial.value,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([5, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
        )
        np.testing.assert_almost_equal(axis.euler_angles.value,
                                       desired=np.deg2rad(np.array([0, 5, 0])), decimal=5)

    def test_set_axis_phi_specified_value_pint(self):
        axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            origin=ValidOrigins.Inertial.value,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([5, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
        )
        axis.phi = np.array([4, ]) * ureg.degree
        np.testing.assert_almost_equal(axis.euler_angles.value,
                                       desired=np.deg2rad(np.array([4, 5, 0])), decimal=5)

    def test_set_axis_psi_specified_value_csdl(self):
        axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            origin=ValidOrigins.Inertial.value,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([5, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
        )
        axis.psi.set_value(np.array([np.deg2rad(-3.)]))
        np.testing.assert_almost_equal(axis.euler_angles.value,
                                       desired=np.deg2rad(np.array([0, 5, -3])), decimal=5)
