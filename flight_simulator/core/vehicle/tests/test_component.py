from pathlib import Path
from unittest import TestCase

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component, ComponentQuantities, ComponentParameters
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.vehicle.controls.aircraft_control_system import AircraftControlSystem


class TestComponentInitialization(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_default_initialization(self):
        """Test default initialization of a Component."""
        component = Component(name="custom_component")
        self.assertEqual(component._name, "custom_component")
        self.assertIsNone(component.geometry)
        self.assertIsInstance(component.quantities, ComponentQuantities)
        self.assertIsInstance(component.parameters, ComponentParameters)
        self.assertEqual(component.comps, {})
        self.assertIsNone(component.quantities.surface_area)
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

 


class TestComponentGeometry(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        from flight_simulator import GEOMETRIES_ROOT_FOLDER
        self.wing_geometry = import_geometry("simple_wing.stp", file_path=GEOMETRIES_ROOT_FOLDER / 'test_geometries')

    def test_geometry_surface_area(self):
        wing_component = Component(name="Wing",
                                   geometry=self.wing_geometry,
                                   compute_surface_area_flag=True)
        np.testing.assert_almost_equal(wing_component.quantities.surface_area.value, 97.229, decimal=3)


class TestComponentHierarchy(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_adding_subcomponent(self):
        """Test adding a subcomponent to a Component."""
        parent_component = Component(name="parent_component")
        sub_component = Component(name="sub_component")

        parent_component.add_subcomponent(sub_component)

        self.assertIn("sub_component", parent_component.comps)
        self.assertEqual(parent_component.comps["sub_component"], sub_component)
        self.assertEqual(sub_component.parent, parent_component)

    def test_adding_subcomponent_that_already_exists(self):
        """Test adding a subcomponent that already exists in the parent Component."""
        parent_component = Component(name="parent_component")
        sub_component = Component(name="sub_component")

        parent_component.add_subcomponent(sub_component)

        with self.assertRaises(KeyError):
            parent_component.add_subcomponent(sub_component)

    def test_removing_subcomponent(self):
        """Test removing a subcomponent from a Component."""
        parent_component = Component(name="parent_component")
        sub_component1 = Component(name="sub_component1")
        sub_component2 = Component(name="sub_component2")

        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)

        parent_component.remove_subcomponent(sub_component1)

        self.assertNotIn("sub_component1", parent_component.comps)
        self.assertIsNone(sub_component1.parent)
        self.assertIn("sub_component2", parent_component.comps)

    def test_visualize_component_hierarchy(self):
        """Test removing a subcomponent from a Component."""
        parent_component = Component(name="parent_component")
        sub_component1 = Component(name="sub_component1")
        sub_component2 = Component(name="sub_component2")

        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)

        parent_component.visualize_component_hierarchy(file_name="component_hierarchy",file_format="png", filepath=Path.cwd()/"python_test_outputs")
        self.assertTrue((Path.cwd()/"python_test_outputs/component_hierarchy.png").exists())
        # Delete the folder and all its files after the test
        (Path.cwd()/"python_test_outputs/component_hierarchy.png").unlink()
        (Path.cwd()/"python_test_outputs/component_hierarchy").unlink()
        (Path.cwd() / "python_test_outputs/").rmdir()




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
