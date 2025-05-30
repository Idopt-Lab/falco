from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator import REPO_ROOT_FOLDER, Q_, ureg
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector
import csdl_alpha as csdl


def add_mp_to_components(aircraft_component: Component, geo_dict: dict, axis_dict: dict):

    # region Wing
    aircraft_component.comps['Wing'].mass_properties = MassProperties(
        mass=Q_(152.88, 'kg'),
        cg=Vector(vector=Q_(geo_dict['wing_le_center'].value, 'm'),
                  axis=axis_dict['wing_axis']),
        inertia=MassMI(axis=axis_dict['wing_axis'])
    )

    aircraft_component.comps['Wing'].comps['left_aileron'].mass_properties = MassProperties(
        mass=Q_(0.1, 'kg'),
        cg=Vector(vector=Q_(geo_dict['left_aileron_le_center'].value, 'm'),
                  axis=axis_dict['left_aileron_axis']),
        inertia=MassMI(axis=axis_dict['left_aileron_axis'])
    )

    right_aileron.mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                                   cg=Vector(vector=Q_(geo_dict['right_aileron_le_center'].value, 'm'),
                                                             axis=right_aileron_axis),
                                                   inertia=MassMI(axis=right_aileron_axis))

    left_flap.mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                               cg=Vector(vector=Q_(geo_dict['left_flap_le_center'].value, 'm'),
                                                         axis=left_flap_axis), inertia=MassMI(axis=left_flap_axis))
    right_flap.mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                                cg=Vector(vector=Q_(geo_dict['right_flap_le_center'].value, 'm'),
                                                          axis=right_flap_axis), inertia=MassMI(axis=right_flap_axis))
    # endregion

    # region Fuselage
    fuselage_comp.mass_properties = MassProperties(mass=Q_(235.87, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([0, 0, 2.6]), 'm'), axis=wing_axis),
                                                   inertia=MassMI(axis=wing_axis))

    battery.mass_properties = MassProperties(mass=Q_(390.08, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([0.1, 0, 2.6]), 'm'), axis=wing_axis),
                                             inertia=MassMI(axis=wing_axis))
    landing_gear.mass_properties = MassProperties(mass=Q_(61.15, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([0, 0, 2.6]), 'm'), axis=wing_axis),
                                                  inertia=MassMI(axis=wing_axis))
    pilots.mass_properties = MassProperties(mass=Q_(170, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([1, 0, 2]), 'm'), axis=wing_axis),
                                            inertia=MassMI(axis=wing_axis))
    miscellaneous.mass_properties = MassProperties(mass=Q_(135.7, 'kg'), cg=Vector(
        vector=Q_(geo_dict['fuselage_wing_qc'].value + np.array([1, 0, 2.6]), 'm'), axis=wing_axis),
                                                   inertia=MassMI(axis=wing_axis))

    # endregion

    # region Empennage
    hor_tail.mass_properties = MassProperties(mass=Q_(27.3 / 2, 'kg'),
                                              cg=Vector(vector=Q_(geo_dict['ht_le_center'].value, 'm'),
                                                        axis=ht_tail_axis), inertia=MassMI(axis=ht_tail_axis))
    trim_tab.mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                              cg=Vector(vector=Q_(geo_dict['trimTab_le_center'].value, 'm'),
                                                        axis=trimTab_axis), inertia=MassMI(axis=trimTab_axis))

    vert_tail.mass_properties = MassProperties(mass=Q_(27.3 / 2, 'kg'),
                                               cg=Vector(vector=Q_(geo_dict['vt_le_mid'].value, 'm'),
                                                         axis=vt_tail_axis), inertia=MassMI(axis=vt_tail_axis))
    rudder.mass_properties = MassProperties(mass=Q_(0.1, 'kg'),
                                            cg=Vector(vector=Q_(geo_dict['rudder_le_mid'].value, 'm'),
                                                      axis=rudder_axis), inertia=MassMI(axis=rudder_axis))
    # endregion


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

    aircraft_component.compute_total_mass_properties()

    aircraft_component.compute_total_loads()


    recorder.stop()
