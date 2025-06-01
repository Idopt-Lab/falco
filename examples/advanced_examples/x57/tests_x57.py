from flight_simulator import REPO_ROOT_FOLDER, Q_
import sys
import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.vehicle.conditions import aircraft_conditions

x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))
from x57_geometry import get_geometry, get_geometry_related_axis
from x57_component import build_aircraft_component
from x57_mp import add_mp_to_components
from x57_control_system import X57ControlSystem
from x57_solvers import X57Aerodynamics, X57Propulsion, HLPropCurve, CruisePropCurve



debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
recorder.start()



# region Mass properties tests
# Mass properties alone test
# - Loads due to mass properties at 0 pitch angle
# - Loads due to mass properties at non-0 pitch angle



geometry_data = get_geometry()
axis_dict = get_geometry_related_axis(geometry_data)
aircraft_component = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)

add_mp_to_components(aircraft_component, geometry_data, axis_dict)

aircraft_component.mass_properties = aircraft_component.compute_total_mass_properties()

x57_controls = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                hl_engine_count=12,cm_engine_count=2)

cruise1 = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(2500, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(100, 'm/s'),
    pitch_angle=Q_(0, 'deg'))

cruise2 = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(2500, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(100, 'm/s'),
    pitch_angle=Q_(2, 'deg'))

tf1, tm1 = aircraft_component.compute_total_loads(fd_state=cruise1.ac_states,controls=x57_controls)
print("Total inertial loads at 0 pitch angle:")
print("Force:", tf1.value)
print("Moment:", tm1.value)

tf2, tm2 = aircraft_component.compute_total_loads(fd_state=cruise2.ac_states,controls=x57_controls)
print("Total inertial loads at 2 pitch angle:")
print("Force:", tf2.value)
print("Moment:", tm2.value)

# endregion

# region Aero Test
# Aero alone test
# - Aero produces loads at wind axis -> wing axis -> FD body-fixed axis
# - L/D when cruise motors are off < L/D when cruise motors are on
# - CL when HLP are off << CL when HLP are on



aircraft_component2 = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)

add_mp_to_components(aircraft_component2, geometry_data, axis_dict)

aircraft_component2.mass_properties = aircraft_component2.compute_total_mass_properties()

x57_controlsA = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                hl_engine_count=12,cm_engine_count=2)

for left_engine, right_engine in zip(x57_controlsA.hl_engines_left, x57_controlsA.hl_engines_right):
    left_engine.throttle.value = 0.0
    right_engine.throttle.value=0.0

x57_controlsB = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                hl_engine_count=12,cm_engine_count=2)

cruise3A = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controlsA,
    altitude=Q_(2500, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(100, 'm/s'),
    pitch_angle=Q_(0, 'deg'))

cruise3B = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controlsB,
    altitude=Q_(2500, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(100, 'm/s'),
    pitch_angle=Q_(0, 'deg'))

print('Wing Axis:',aircraft_component2.comps['Wing'].mass_properties.cg_vector.vector.value)

x57_aerodynamics = X57Aerodynamics(component=aircraft_component2)
aircraft_component2.comps['Wing'].load_solvers.append(x57_aerodynamics)

tf3A, tm3A = aircraft_component2.compute_total_loads(fd_state=cruise3A.ac_states, controls=x57_controlsA)
tf3B, tm3B = aircraft_component2.compute_total_loads(fd_state=cruise3B.ac_states, controls=x57_controlsB)

tf3A = tf3A - tf1
tm3A = tm3A - tm1
print("Total aero loads and aero loads alone, with HLP off:")
print("Force:", tf3A.value)
print("Moment:", tm3A.value)

tf3B = tf3B - tf1
tm3B = tm3B - tm1
print("Total aero loads and aero loads alone, with HLP on:")
print("Force:", tf3B.value)
print("Moment:", tm3B.value)



# endregion





# region Propulsion Test
# Prop alone test
# - If CM axis location is offset from FD axis, there is a pitch moment
# - CM left is on; CM right is off. There is a yaw moment
# - When both CM are ok; there is no yaw moment




aircraft_component3 = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)
aircraft_component4 = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)
aircraft_component5 = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)


add_mp_to_components(aircraft_component3, geometry_data, axis_dict)
add_mp_to_components(aircraft_component4, geometry_data, axis_dict)
add_mp_to_components(aircraft_component5, geometry_data, axis_dict)


aircraft_component3.mass_properties = aircraft_component3.compute_total_mass_properties()
aircraft_component4.mass_properties = aircraft_component4.compute_total_mass_properties()
aircraft_component5.mass_properties = aircraft_component5.compute_total_mass_properties()


x57_controls = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                hl_engine_count=0,cm_engine_count=2)

cruise4 = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(2500, 'ft'),
    range=Q_(100, 'km'),
    speed=Q_(44.08789, 'm/s'),
    pitch_angle=Q_(0, 'deg'))


cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft

print('Cruise Motor 1 CG Location:',aircraft_component3.comps['Wing'].comps['Cruise Motor 1'].mass_properties.cg_vector.vector.value)
print('Cruise Motor 2 CG Location:',aircraft_component3.comps['Wing'].comps['Cruise Motor 2'].mass_properties.cg_vector.vector.value)

cruise_motor1_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=0)
cruise_motor2_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=1)

aircraft_component3.comps['Wing'].comps['Cruise Motor 1'].load_solvers.append(cruise_motor1_prop)
aircraft_component4.comps['Wing'].comps['Cruise Motor 2'].load_solvers.append(cruise_motor2_prop)

aircraft_component5.comps['Wing'].comps['Cruise Motor 1'].load_solvers.append(cruise_motor1_prop)
aircraft_component5.comps['Wing'].comps['Cruise Motor 2'].load_solvers.append(cruise_motor2_prop)



tf4, tm4 = aircraft_component3.compute_total_loads(fd_state=cruise4.ac_states, controls=x57_controls)
tf5, tm5 = aircraft_component4.compute_total_loads(fd_state=cruise4.ac_states, controls=x57_controls)
tf6, tm6 = aircraft_component5.compute_total_loads(fd_state=cruise4.ac_states, controls=x57_controls)


tf4 = tf4 - tf1
tm4 = tm4 - tm1
print("Total prop loads and prop loads alone with only CM1 prop:")
print("Force:", tf4.value)
print("Moment:", tm4.value)

tf5 = tf5 - tf1
tm5 = tm5 - tm1
print("Total prop loads and prop loads alone with only CM2 prop:")
print("Force:", tf5.value)
print("Moment:", tm5.value)

tf6 = tf6 - tf1
tm6 = tm6 - tm1
print("Total prop loads and prop loads alone with both CM1 and CM2 prop:")
print("Force:", tf6.value)
print("Moment:", tm6.value)


for left_engine, right_engine in zip(x57_controlsA.cm_engines_left, x57_controlsA.cm_engines_right):
    left_engine.throttle.value = 0.0
    right_engine.throttle.value=0.0

cruise4B = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(2500, 'ft'),
    range=Q_(100, 'km'),
    speed=Q_(44.08789, 'm/s'),
    pitch_angle=Q_(0, 'deg'))

tf7, tm7 = aircraft_component5.compute_total_loads(fd_state=cruise4B.ac_states, controls=x57_controls)
tf7 = tf7 - tf1
tm7 = tm7 - tm1
print("Total prop loads and prop loads alone with both CM1 and CM2 prop, with engines off:")
print("Force:", tf7.value)
print("Moment:", tm7.value)

# endregion





# todo: after Aviation paper
# - Test parameterization solver when aircraft components are defined in a heirarchy