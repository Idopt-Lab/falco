from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator import REPO_ROOT_FOLDER, Q_, ureg
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector

import csdl_alpha as csdl
import numpy as np


def add_mp_to_components(aircraft_component: Component, geo_dict: dict, axis_dict: dict):


    # region Wing
    aircraft_component.comps['Wing'].mass_properties = MassProperties(
        mass=Q_(152.88, 'kg'),
        cg=Vector(vector=Q_(geo_dict['wing_le_center'].value, 'm'),
                  axis=axis_dict['wing_axis']),
        inertia=MassMI(axis=axis_dict['wing_axis'])
    )

    aircraft_component.comps['Wing'].comps['Left Aileron'].mass_properties = MassProperties(
        mass=Q_(0.1, 'kg'),
        cg=Vector(vector=Q_(geo_dict['left_aileron_le_center'].value, 'm'),
                  axis=axis_dict['left_aileron_axis']),
        inertia=MassMI(axis=axis_dict['left_aileron_axis'])
    )

    aircraft_component.comps['Wing'].comps['Right Aileron'].mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                                   cg=Vector(vector=Q_(geo_dict['right_aileron_le_center'].value, 'm'),
                                                             axis=axis_dict['right_aileron_axis']),
                                                   inertia=MassMI(axis=axis_dict['right_aileron_axis']))

    aircraft_component.comps['Wing'].comps['Left Flap'].mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                               cg=Vector(vector=Q_(geo_dict['left_flap_le_center'].value, 'm'),
                                                         axis=axis_dict['left_flap_axis']), inertia=MassMI(axis=axis_dict['left_flap_axis']))
    aircraft_component.comps['Wing'].comps['Right Flap'].mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                                cg=Vector(vector=Q_(geo_dict['right_flap_le_center'].value, 'm'),
                                                          axis=axis_dict['right_flap_axis']), inertia=MassMI(axis=axis_dict['right_flap_axis']))
    # endregion

    # region Fuselage
    aircraft_component.comps['Fuselage'].mass_properties = MassProperties(mass=Q_(235.87, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([0, 0, 2.6]), 'm'), axis=axis_dict['wing_axis']),
                                                   inertia=MassMI(axis=axis_dict['wing_axis']))

    aircraft_component.comps['Fuselage'].comps['Battery'].mass_properties = MassProperties(mass=Q_(390.08, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([0.1, 0, 2.6]), 'm'), axis=axis_dict['wing_axis']),
                                             inertia=MassMI(axis=axis_dict['wing_axis']))
    aircraft_component.comps['Fuselage'].comps['Landing Gear'].mass_properties = MassProperties(mass=Q_(61.15, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([0, 0, 2.6]), 'm'), axis=axis_dict['wing_axis']),
                                                  inertia=MassMI(axis=axis_dict['wing_axis']))
    aircraft_component.comps['Fuselage'].comps['Pilots'].mass_properties = MassProperties(mass=Q_(170, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([1, 0, 2]), 'm'), axis=axis_dict['wing_axis']),
                                            inertia=MassMI(axis=axis_dict['wing_axis']))
    aircraft_component.comps['Fuselage'].comps['Miscellaneous'].mass_properties = MassProperties(mass=Q_(135.7, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([1, 0, 2.6]), 'm'), axis=axis_dict['wing_axis']),
                                                   inertia=MassMI(axis=axis_dict['wing_axis']))

    # endregion

    # region Empennage
    aircraft_component.comps['Elevator'].mass_properties = MassProperties(mass=Q_(27.3 / 2, 'kg'),
                                              cg=Vector(vector=Q_(geo_dict['ht_le_center'].value - np.array([0,0.1965899,0]), 'm'),
                                                        axis=axis_dict['ht_tail_axis']), inertia=MassMI(axis=axis_dict['ht_tail_axis']))
    aircraft_component.comps['Elevator'].comps['Trim Tab'].mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                              cg=Vector(vector=Q_(geo_dict['trimTab_le_center'].value, 'm'),
                                                        axis=axis_dict['trim_tab_axis']), inertia=MassMI(axis=axis_dict['trim_tab_axis']))

    aircraft_component.comps['Vertical Tail'].mass_properties = MassProperties(mass=Q_(27.3 / 2, 'kg'),
                                               cg=Vector(vector=Q_(geo_dict['vt_le_mid'].value, 'm'),
                                                         axis=axis_dict['vt_tail_axis']), inertia=MassMI(axis=axis_dict['vt_tail_axis']))
    aircraft_component.comps['Vertical Tail'].comps['Rudder'].mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                            cg=Vector(vector=Q_(geo_dict['rudder_le_mid'].value, 'm'),
                                                      axis=axis_dict['rudder_axis']), inertia=MassMI(axis=axis_dict['rudder_axis']))
    # endregion

    # region Propulsion

    hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

    for i, hl_motor in enumerate(hl_motors):
        hl_motor.mass_properties = MassProperties(
            mass=Q_(81.65/12, 'kg'),
            cg=Vector(vector=Q_(geo_dict['MotorDisks'][i].value, 'm'),
                    axis=axis_dict['hl_motor_axes'][i]),
            inertia=MassMI(axis=axis_dict['hl_motor_axes'][i])
        )
    
    cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

    for i, cruise_motor in enumerate(cruise_motors):
        cruise_motor.mass_properties = MassProperties(
            mass=Q_(106.14/2, 'kg'),
            cg=Vector(vector=Q_(geo_dict['cruise_motors_base'][i].value, 'm'),
                       axis=axis_dict['cruise_motor_axes'][i]),
            inertia=MassMI(axis=axis_dict['cruise_motor_axes'][i])
        )
    
    # endregion


    aircraft_component.mass_properties = MassProperties(
        mass=Q_(0, 'kg'),
        cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=axis_dict['fd_axis']),
        inertia=MassMI(axis=axis_dict['fd_axis'])
    )

    return


if __name__ == "__main__":
    from flight_simulator import REPO_ROOT_FOLDER
    import sys

    x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
    sys.path.append(str(x57_folder_path))
    from x57_geometry import get_geometry, get_geometry_related_axis
    from x57_component import build_aircraft_component

    debug = False
    recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
    recorder.start()

    geometry_data = get_geometry()
    axis_dict = get_geometry_related_axis(geometry_data)
    aircraft_component = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)

    add_mp_to_components(aircraft_component, geometry_data, axis_dict)

    aircraft_component.mass_properties = aircraft_component.compute_total_mass_properties()
    print(repr(aircraft_component))

    for comp in aircraft_component.comps['Wing'].comps.values():
        if hasattr(comp, 'mass_properties'):
            print(f"{comp._name} mass properties: {comp.mass_properties.cg_vector.vector.value}")

    # aircraft_component.compute_total_loads()


    recorder.stop()
