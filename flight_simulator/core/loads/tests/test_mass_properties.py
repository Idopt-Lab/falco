from unittest import TestCase

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component

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




class TestComponentMassProperties(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )
    

        self.parent_component = Component(name="Aircraft")
        self.sub_component1 = Component(name="Wing")
        self.sub_component2 = Component(name="Motor")
        self.sub_component3 = Component(name="Fuselage")


        # Add subcomponents to the parent component
        self.parent_component.add_subcomponent(self.sub_component1)
        self.parent_component.add_subcomponent(self.sub_component2)
        self.parent_component.add_subcomponent(self.sub_component3)

    def test_mass_prop_buildup(self):
        """Test providing mass properties during initialization."""

        # Create mass properties for the parent component
        self.parent_component.quantities.mass_properties.mass = Q_(0, 'kg')
        self.parent_component.quantities.mass_properties.cg_vector = Vector(Q_([1, 0, 0], 'm'), self.axis)

        # Create mass properties for the first subcomponent
        self.sub_component1.quantities.mass_properties.mass = Q_(15, 'kg')
        self.sub_component1.quantities.mass_properties.cg_vector = Vector(Q_([0, 1, 0], 'm'), self.axis)

        # Create mass properties for the second subcomponent
        self.sub_component2.quantities.mass_properties.mass = Q_(5, 'kg')
        self.sub_component2.quantities.mass_properties.cg_vector = Vector(Q_([0, 0, 1], 'm'), self.axis)

        # Create mass properties for the third subcomponent
        self.sub_component3.quantities.mass_properties.mass = Q_(20, 'kg')
        self.sub_component3.quantities.mass_properties.cg_vector = Vector(Q_([0, 1, 0], 'm'), self.axis)

        # Recursively compute mass properties from the parent component
        total_mp = self.parent_component.compute_mass_properties()


        # Compute the total mass (as a Q_ quantity)
        total_mass = (self.parent_component.quantities.mass_properties.mass +
                    self.sub_component1.quantities.mass_properties.mass +
                    self.sub_component2.quantities.mass_properties.mass +
                    self.sub_component3.quantities.mass_properties.mass)

        # Compute the weighted sum of the cg vectors. We assume that each cg_vector has an attribute
        expected_cg = (
            self.parent_component.quantities.mass_properties.mass.magnitude * np.array(self.parent_component.quantities.mass_properties.cg_vector.vector.value) +
            self.sub_component1.quantities.mass_properties.mass.magnitude * np.array(self.sub_component1.quantities.mass_properties.cg_vector.vector.value) +
            self.sub_component2.quantities.mass_properties.mass.magnitude * np.array(self.sub_component2.quantities.mass_properties.cg_vector.vector.value) +
            self.sub_component3.quantities.mass_properties.mass.magnitude * np.array(self.sub_component3.quantities.mass_properties.cg_vector.vector.value)
        ) / total_mass.magnitude


        # Total mass should be the sum of the parent's and the subcomponents' masses
        np.testing.assert_almost_equal(total_mp.mass.value, 40)
        # Total CG should be the weighted average of the CGs of the parent and subcomponents
        np.testing.assert_almost_equal(total_mp.cg_vector.vector.value, expected_cg)
        print("Total CG:", total_mp.cg_vector.vector.value)
