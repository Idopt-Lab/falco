from unittest import TestCase

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.aircraft.component import Component, ComponentQuantities, ComponentParameters


class TestComponentInitialization(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.axis = Axis(
            name='Inertial Axis',
            translation=np.array([0, 0, 0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create a Quantity vector with units
        cg = Q_([0, 0, 0], 'newton')
        cg = Vector(cg, self.axis)
        # Create a mass moment of inertia object
        mi = MassMI(axis=self.axis)
        # Create a mass properties object
        self.mp = MassProperties(cg_vector=cg, inertia_tensor=mi, mass=Q_(10, 'lb'))

    def test_default_initialization(self):
        """Test default initialization of a Component."""
        component = Component(name="custom_component")
        self.assertEqual(component._name, "custom_component")
        self.assertIsNone(component.geometry)
        self.assertIsInstance(component.quantities, ComponentQuantities)
        self.assertIsInstance(component.parameters, ComponentParameters)
        self.assertEqual(component.comps, {})
        self.assertIsNone(component.surface_area)
        self.assertFalse(component.compute_surface_area_flag)

    def test_kwargs_parameters(self):
        """Test initialization with user-defined parameters using kwargs."""
        component = Component(name="custom_component",
                              param1="value1", param2=123)
        self.assertEqual(component.parameters.param1, "value1")
        self.assertEqual(component.parameters.param2, 123)

    def test_parent_attribute(self):
        """Test the parent attribute initialization."""
        component = Component(name="custom_component")
        self.assertIsNone(component.parent)

    def test_empty_subcomponents(self):
        """Test that the subcomponents dictionary is empty upon initialization."""
        component = Component(name="custom_component")
        self.assertEqual(component.comps, {})

    def test_providing_mp(self):
        component = Component(name="custom_component")
        component.quantities.mass_properties = self.mp
        np.testing.assert_equal(component.quantities.mass_properties.inertia_tensor.inertia_tensor.value,
                                np.zeros((3, 3)))
        np.testing.assert_almost_equal(component.quantities.mass_properties.mass.value, desired=4.53592, decimal=5)