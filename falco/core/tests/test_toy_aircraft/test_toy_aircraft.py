from pathlib import Path
from unittest import TestCase

from falco import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from falco.core.tests.test_toy_aircraft.toy_aircraft_models import ToyAircraftControlSystem


def get_geometry(inertial_axis):
    from falco.core.vehicle.components.aircraft import Aircraft
    from falco.core.vehicle.components.component import Component, ComponentParameters
    from falco.core.loads.mass_properties import MassProperties, MassMI
    from falco.core.dynamics.axis import Axis, ValidOrigins
    from falco.core.loads.forces_moments import Vector

    aircraft_component = Aircraft()

    # region Fuselage Component
    fuselage_component = Component(name='Fuselage')
    aircraft_component.add_subcomponent(fuselage_component)

    # region Engine Component
    engine_axis = Axis(
        name='Engine Axis',
        x=Q_(-5, 'ft'),
        y=Q_(0, 'ft'),
        z=Q_(0, 'ft'),
        phi=Q_(0, 'deg'),
        theta=Q_(5, 'deg'),  # Engine is pitched up by 5 deg
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    engine_component = Component(name='Engine')
    engine_mass_properties = MassProperties(mass=Q_(100, 'kg'),
                                            inertia=MassMI(axis=engine_axis,
                                                           Ixx=Q_(0, 'kg*(m*m)'),
                                                           Iyy=Q_(0, 'kg*(m*m)'),
                                                           Izz=Q_(0, 'kg*(m*m)'),
                                                           Ixy=Q_(0, 'kg*(m*m)'),
                                                           Ixz=Q_(0, 'kg*(m*m)'),
                                                           Iyz=Q_(0, 'kg*(m*m)')),
                                            cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'),
                                                      axis=engine_axis))
    engine_component.mass_properties = engine_mass_properties
    fuselage_component.add_subcomponent(engine_component)
    # endregion

    # endregion

    # region Wing Component

    wing_axis = Axis(
        name='Wing Axis',
        x=Q_(0, 'ft'),
        y=Q_(0, 'ft'),
        z=Q_(0, 'ft'),  # z is positive down in FD axis
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),  # This is incidence angle of the wing
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )
    wing_mass_properties = MassProperties(mass=Q_(500, 'kg'),
                                          inertia=MassMI(axis=wing_axis,
                                                         Ixx=Q_(0, 'kg*(m*m)'),
                                                         Iyy=Q_(0, 'kg*(m*m)'),
                                                         Izz=Q_(0, 'kg*(m*m)'),
                                                         Ixy=Q_(0, 'kg*(m*m)'),
                                                         Ixz=Q_(0, 'kg*(m*m)'),
                                                         Iyz=Q_(0, 'kg*(m*m)')),
                                          cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'),
                                                    axis=wing_axis))

    wing_component = Component(name='Wing')
    wing_component.mass_properties = wing_mass_properties
    aircraft_component.add_subcomponent(wing_component)
    # endregion
    return aircraft_component

def get_control_system():
    control_system = ToyAircraftControlSystem()

    return control_system


def get_solvers():
    from falco.core.tests.test_toy_aircraft.toy_aircraft_models import ToyAircraftPropulsion, ToyAircraftAerodynamics



    return


class TestComponent(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        from falco.core.dynamics.axis import Axis, ValidOrigins

        self.inertial_axis = Axis(
            name='Inertial Axis',
            origin=ValidOrigins.Inertial.value
        )

    def test_create_aircraft_component(self):
        aircraft = get_geometry(self.inertial_axis)

        self.assertEqual(list(aircraft.comps.keys()), ['Fuselage', 'Wing'])
        self.assertEqual(list(aircraft.comps['Fuselage'].comps.keys()), ['Engine'])

        engine_axis = aircraft.comps['Fuselage'].comps['Engine'].mass_properties.cg_vector.axis
        self.assertEqual(engine_axis.name, 'Engine Axis')
        np.testing.assert_almost_equal(engine_axis.translation_from_origin_vector.value,
                                       desired=np.array([-1.524, 0, 0]))
        np.testing.assert_almost_equal(engine_axis.euler_angles_vector.value,
                                       desired=np.deg2rad(np.array([0, 5, 0])))

        self.assertEqual(aircraft.comps['Wing'].mass_properties.mass.value, 500)

    def test_modify_aircraft_component_axis(self):
        aircraft = get_geometry(self.inertial_axis)

        engine_axis = aircraft.comps['Fuselage'].comps['Engine'].mass_properties.cg_vector.axis
        engine_axis.translation_from_origin_vector.set_value(value=np.array([0, 0, -5]))
        engine_axis.euler_angles_vector.set_value(value=np.deg2rad(np.array([0, 10, 0])))

        np.testing.assert_almost_equal(engine_axis.translation_from_origin_vector.value,
                                       desired=np.array([0, 0, -5]))
        np.testing.assert_almost_equal(engine_axis.euler_angles_vector.value,
                                       desired=np.deg2rad(np.array([0, 10, 0])))

    def test_modify_aircraft_component_values(self):
        aircraft = get_geometry(self.inertial_axis)

        self.assertEqual(aircraft.comps['Wing'].mass_properties.mass.value, 500)

        aircraft.comps['Wing'].mass_properties.mass.set_value(value=600)
        self.assertEqual(aircraft.comps['Wing'].mass_properties.mass.value, 600)


class TestControlSystem(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        from falco.core.dynamics.axis import Axis, ValidOrigins

        self.inertial_axis = Axis(
            name='Inertial Axis',
            origin=ValidOrigins.Inertial.value
        )