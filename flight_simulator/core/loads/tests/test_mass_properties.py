from unittest import TestCase

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI

class TestMassProperties(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.axis = Axis(
            name='Inertial Axis',
            x=np.array([0,]) * ureg.meter,
            y=np.array([0,]) * ureg.meter,
            z=np.array([0,]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create a Quantity vector with units
        vector_quantity = Q_([10, 20, 30], 'newton')
        self.vector = Vector(vector_quantity, self.axis)

    def test_mi_initialization(self):
        mi = MassMI(axis=self.axis)
        np.testing.assert_almost_equal(mi.inertia_tensor.value, np.zeros((3,3)))

    def test_mi_initialization_with_quantity(self):
        m = Q_(10, 'lb')
        r = Q_(5, 'ft')
        I = m*r**2
        mi = MassMI(axis=self.axis,
                    Ixx=I, Iyy=I, Izz=I)
        actual = np.zeros((3, 3))
        actual[0, 0] = 0.04214 * (10 * 5 ** 2)
        actual[1, 1] = 0.04214 * (10 * 5 ** 2)
        actual[2, 2] = 0.04214 * (10 * 5 ** 2)
        np.testing.assert_almost_equal(mi.inertia_tensor.value, actual, decimal=3)

    def test_mp_initialization(self):
        mi = MassMI(axis=self.axis)
        mp = MassProperties(cg=self.vector, inertia=mi, mass=Q_(10, 'lb'))
        np.testing.assert_almost_equal(actual=mp.mass.value, desired=4.53592, decimal=5)
