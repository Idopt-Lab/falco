import csdl_alpha as csdl
import numpy as np

from falco.core.vehicle.components.component import Component
from falco.core.vehicle.components.wing import Wing as WingComp
from falco.core.vehicle.components.fuselage import Fuselage as FuseComp
from falco.core.vehicle.components.aircraft import Aircraft as AircraftComp
from falco.core.vehicle.components.rotor import Rotor as RotorComp
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables



# Aircraft Component Creation
def build_aircraft_component(geo_dict: dict, do_geo_param: bool = False):
    geometry = geo_dict['geometry']
    parameterization_solver = ParameterizationSolver()
    ffd_geometric_variables = GeometricVariables()

    # region Top-level Aircraft Component
    if do_geo_param:
        aircraft_component = AircraftComp(geometry=geometry, compute_surface_area_flag=False,
                                            parameterization_solver=parameterization_solver,
                                            ffd_geometric_variables=ffd_geometric_variables)
    else:
        aircraft_component = AircraftComp(geometry=geometry)

    # endregion

    # region Fuselage components
    if do_geo_param:
        fuselage_comp = FuseComp(
            length=csdl.Variable(name="fuselage_length", shape=(1,), value=8.2242552),
            max_height=csdl.Variable(name="fuselage_max_height", shape=(1,), value=1.09),
            max_width=csdl.Variable(name="fuselage_max_width", shape=(1,), value=1.24070602),
            geometry=geo_dict['fuselage'], skip_ffd=False,
            parameterization_solver=parameterization_solver,
            ffd_geometric_variables=ffd_geometric_variables)
    else:
        fuselage_comp = FuseComp(
            length=csdl.Variable(name="fuselage_length", shape=(1,), value=8.2242552),
            max_height=csdl.Variable(name="fuselage_max_height", shape=(1,), value=1.09),
            max_width=csdl.Variable(name="fuselage_max_width", shape=(1,), value=1.24070602),
            geometry=geo_dict['fuselage'], skip_ffd=True)
    aircraft_component.add_subcomponent(fuselage_comp)
    # endregion

    # region Wing components

    wing_AR = csdl.Variable(name="wing_AR", shape=(1,), value=15)
    wing_span = csdl.Variable(name="wing_span", shape=(1,), value=9.6)
    wing_sweep = csdl.Variable(name="wing_sweep", shape=(1,), value=0)
    wing_dihedral = csdl.Variable(name="wing_dihedral", shape=(1,), value=0)
    if do_geo_param:
        wing = WingComp(AR=wing_AR,
                        span=wing_span,
                        sweep=wing_sweep,
                        dihedral=wing_dihedral,
                        geometry=geo_dict['wingALL'],
                        parametric_geometry=geo_dict['wing_parametric_geometry'],
                        skip_ffd=False,
                        tight_fit_ffd=False,
                        orientation='horizontal',
                        name='Wing', 
                        parameterization_solver=parameterization_solver,
                        ffd_geometric_variables=ffd_geometric_variables
                        )
    else:
        wing = WingComp(AR=wing_AR,
                        span=wing_span,
                        sweep=wing_sweep,
                        dihedral=wing_dihedral,
                        geometry=geo_dict['wingALL'],
                        skip_ffd=True,
                        orientation='horizontal',
                        name='Wing')
    wing.parameters.actuate_angle = csdl.Variable(name="wing_incidence", shape=(1,),
                                                  value=np.deg2rad(2))  # Wing incidence angle in radians
    aircraft_component.add_subcomponent(wing)

    if do_geo_param:
        wing_qc_fuse_connection = geometry.evaluate(geo_dict['wing_qc_center_parametric']) - geometry.evaluate(
            geo_dict['fuselage_wing_qc_center_parametric'])
        parameterization_solver.add_variable(computed_value=wing_qc_fuse_connection,
                                             desired_value=wing_qc_fuse_connection.value)

    # Left aileron
    left_aileron = Component(name='Left Aileron')
    left_aileron.parameters.actuate_angle = csdl.Variable(name="left_aileron_actuate_angle", shape=(1,),
                                                          value=np.deg2rad(0))
    wing.add_subcomponent(left_aileron)

    # Right aileron
    right_aileron = Component(name='Right Aileron')
    right_aileron.parameters.actuate_angle = csdl.Variable(name="right_aileron_actuate_angle", shape=(1,),
                                                           value=np.deg2rad(0))
    wing.add_subcomponent(right_aileron)

    # Left flap
    left_flap = Component(name='Left Flap')
    left_flap.parameters.actuate_angle = csdl.Variable(name="left_flap_actuate_angle", shape=(1,), value=np.deg2rad(0))
    wing.add_subcomponent(left_flap)

    # Right flap
    right_flap = Component(name='Right Flap')
    right_flap.parameters.actuate_angle = csdl.Variable(name="right_flap_actuate_angle", shape=(1,),
                                                        value=np.deg2rad(0))
    wing.add_subcomponent(right_flap)
    # endregion

    # region Empennage components

    # Horizontal Tail
    # ht_area = geo_dict['ht_span'] * geo_dict['ht_chord']
    # ht_ar = geo_dict['ht_span'] ** 2 / ht_area
    ht_ar = csdl.Variable(name="HT_AR", shape=(1,), value=4)
    ht_span = csdl.Variable(name="HT_span", shape=(1,), value=3.14986972)
    ht_sweep = csdl.Variable(name="HT_sweep", shape=(1,), value=0)

    if do_geo_param:
        hor_tail = WingComp(AR=ht_ar, span=ht_span, sweep=ht_sweep,
                            geometry=geo_dict['htALL'], parametric_geometry=geo_dict['ht_parametric_geometry'],
                            tight_fit_ffd=False, skip_ffd=False,
                            name='Elevator', orientation='horizontal',
                            parameterization_solver=parameterization_solver,
                            ffd_geometric_variables=ffd_geometric_variables)
    else:
        hor_tail = WingComp(AR=ht_ar, span=ht_span, sweep=ht_sweep,
                            geometry=geo_dict['htALL'],
                            skip_ffd=True,
                            name='Elevator', orientation='horizontal')
        
    aircraft_component.add_subcomponent(hor_tail)
    hor_tail.parameters.actuate_angle = csdl.Variable(name="elevator_actuate_angle", shape=(1,), value=np.deg2rad(0))

    trim_tab = Component(name='Trim Tab')
    trim_tab.parameters.actuate_angle = csdl.Variable(name="Trim Tab Actuate Angle", shape=(1,), value=np.deg2rad(0))
    hor_tail.add_subcomponent(trim_tab)

    if do_geo_param:
        h_tail_fuselage_connection = geometry.evaluate(geo_dict['ht_te_center_parametric']) - geometry.evaluate(
            geo_dict['fuselage_tail_te_center_parametric'])
        parameterization_solver.add_variable(computed_value=h_tail_fuselage_connection,
                                             desired_value=h_tail_fuselage_connection.value)

    # Vertical Tail
    vt_ar = csdl.Variable(name="VT_AR", shape=(1,), value=1.998)
    vt_span = csdl.Variable(name="VT_span", shape=(1,), value=1.6191965383361169)
    vt_sweep = csdl.Variable(name="VT_sweep", shape=(1,), value=-30)

    if do_geo_param:
        vert_tail = WingComp(AR=vt_ar, span=vt_span, sweep=vt_sweep,
                             geometry=geo_dict['vtALL'], parametric_geometry=geo_dict['vt_parametric_geometry'],
                             tight_fit_ffd=False, skip_ffd=False,
                             name='Vertical Tail', orientation='vertical',
                             parameterization_solver=parameterization_solver,
                             ffd_geometric_variables=ffd_geometric_variables)
    else:
        vert_tail = WingComp(AR=vt_ar, span=vt_span, sweep=vt_sweep,
                             geometry=geo_dict['vtALL'],
                             skip_ffd=True,
                             name='Vertical Tail', orientation='vertical')
        
    aircraft_component.add_subcomponent(vert_tail)

    # Rudder
    rudder = Component(name='Rudder')
    rudder.parameters.actuate_angle = csdl.Variable(name="Rudder Actuate Angle", shape=(1,), value=np.deg2rad(0))
    vert_tail.add_subcomponent(rudder)

    if do_geo_param:
        vtail_fuselage_connection = geometry.evaluate(geo_dict['fuselage_rear_pts_parametric']) - geometry.evaluate(
            geo_dict['vt_qc_base_parametric'])
        parameterization_solver.add_variable(computed_value=vtail_fuselage_connection,
                                             desired_value=vtail_fuselage_connection.value)
    # endregion

    # region Rotors
    lift_rotors = []
    for i in range(1, 13):
        hl_motor = RotorComp(name=f'HL Motor {i}', radius=geo_dict['MotorDisks'][i - 1])
        lift_rotors.append(hl_motor)
        wing.add_subcomponent(hl_motor)

    cruise_motors = []
    for i in range(1, 3):
        cruise_motor = RotorComp(name=f'Cruise Motor {i}', radius=geo_dict['cruise_motors_base'][i - 1])
        cruise_motors.append(cruise_motor)
        wing.add_subcomponent(cruise_motor)
    # endregion

    # region Miscellaneous Components
    battery = Component(name='Battery')
    fuselage_comp.add_subcomponent(battery)

    landing_gear = Component(name='Landing Gear')
    fuselage_comp.add_subcomponent(landing_gear)

    pilots = Component(name='Pilots')
    fuselage_comp.add_subcomponent(pilots)

    miscellaneous = Component(name='Miscellaneous')
    fuselage_comp.add_subcomponent(miscellaneous)
    # endregion

    if do_geo_param is True:
        parameterization_solver.evaluate(ffd_geometric_variables)
        geometry.plot(camera=dict(pos=(12, 15, -12),  # Camera position
                                  focal_point=(-fuselage_comp.parameters.length.value / 2, 0, 0),
                                  # Point camera looks at
                                  viewup=(0, 0, -1)),  # Camera up direction
                      title=f'X-57 Maxwell Aircraft Geometry\nWing Span: {wing.parameters.span.value[0]:.2f} m\nWing AR: {wing.parameters.AR.value[0]:.2f}\nWing Area S: {wing.parameters.S_ref.value[0]:.2f} m^2\nWing Sweep: {wing.parameters.sweep.value[0]:.2f} deg',
                      #  title=f'X-57 Maxwell Aircraft Geometry\nFuselage Length: {Fuselage.parameters.length.value[0]:.2f} m\nFuselage Height: {Fuselage.parameters.max_height.value[0]:.2f} m\nFuselage Width: {Fuselage.parameters.max_width.value[0]:.2f} m',
                      screenshot=REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57' / 'images' / f'x_57_{wing.parameters.span.value[0]}_AR_{wing.parameters.AR.value[0]}_S_ref_{wing.parameters.S_ref.value[0]}_sweep_{wing.parameters.sweep.value[0]}.png')

    return aircraft_component


if __name__ == "__main__":
    from falco import REPO_ROOT_FOLDER
    import sys

    x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
    sys.path.append(str(x57_folder_path))
    from x57_geometry import get_geometry

    debug = False
    recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
    recorder.start()

    geometry_data = get_geometry()
    aircraft_component = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)
    aircraft_component._name = 'X-57 Maxwell Aircraft'
    aircraft_component.visualize_component_hierarchy(show=True)
    print("Aircraft component created successfully")

    recorder.stop()
