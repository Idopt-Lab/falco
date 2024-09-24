from unittest import TestCase
from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.utils.axis import Axis


class AxisTests(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_create_axis(self):
        inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            angles=np.array([0, 0, 0]) * ureg.degree,
            origin='inertial'
        )
        np.testing.assert_almost_equal(inertial_axis.translation.value, desired=np.zeros(3,))

    def test_set_axis_translation_via_pint(self):
        inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            angles=np.array([0, 0, 0]) * ureg.degree,
            origin='inertial'
        )
        inertial_axis.translation = np.array([3, 0, 0]) * ureg.ft
        np.testing.assert_almost_equal(inertial_axis.translation.value, desired=np.array([3*0.3048, 0, 0]), decimal=5)

    def test_set_axis_translation_via_csdl(self):
        inertial_axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            angles=np.array([0, 0, 0]) * ureg.degree,
            origin='inertial'
        )
        inertial_axis.translation.set_value(np.array([0, 5, 0]))
        np.testing.assert_almost_equal(inertial_axis.translation.value, desired=np.array([0, 5, 0]))