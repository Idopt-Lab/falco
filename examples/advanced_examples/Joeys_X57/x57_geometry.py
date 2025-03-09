import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
import lsdo_geo as lg
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER
from flight_simulator.core.vehicle.component import Component
from flight_simulator.core.loads.mass_properties import MassProperties
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from typing import Union, List
from dataclasses import dataclass
from flight_simulator import ureg
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.euler_rotations import build_rotation_matrix

lfs.num_workers = 1

debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=debug)

recorder.start()
run_ffd = True
# run_optimization = True


in2m=0.0254
ft2m = 0.3048


geometry = import_geometry(
    file_name="x57.stp",
    file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57',
    refit=False,
    scale=in2m,
    rotate_to_body_fixed_frame=True
)


wing = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4'], name='wing')
aileronR = geometry.declare_component(function_search_names=['Rt_Aileron'], name='aileronR')
aileronL = geometry.declare_component(function_search_names=['Lt_Aileron'], name='aileronL')
flapL = geometry.declare_component(function_search_names=['Flap, 1'], name='left_flap')
flapR = geometry.declare_component(function_search_names=['Flap, 0'], name='right_flap')
wingALL = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4','Rt_Aileron','Lt_Aileron','Flap, 0','Flap, 1'], name='CompleteWing')
h_tail = geometry.declare_component(function_search_names=['HorzStab'], name='h_tail')
trimTab = geometry.declare_component(function_search_names=['TrimTab'], name='trimTab')
htALL = geometry.declare_component(function_search_names=['HorzStab', 'TrimTab'], name='CompleteHT')
vertTail = geometry.declare_component(function_search_names=['VertTail'], name='vertTail')
rudder = geometry.declare_component(function_search_names=['Rudder'], name='rudder')
fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
gear_pod = geometry.declare_component(function_search_names=['GearPod'], name='gear_pod')

pylon1 = geometry.declare_component(function_search_names=['Pylon_07'], name='pylon1')
pylon2 = geometry.declare_component(function_search_names=['Pylon_08'], name='pylon2')
pylon3 = geometry.declare_component(function_search_names=['Pylon_09'], name='pylon3')
pylon4 = geometry.declare_component(function_search_names=['Pylon_10'], name='pylon4')
pylon5 = geometry.declare_component(function_search_names=['Pylon_11'], name='pylon5')
pylon6 = geometry.declare_component(function_search_names=['Pylon_12'], name='pylon6')
pylon7 = geometry.declare_component(function_search_names=['Pylon_07'], name='pylon7')
pylon8 = geometry.declare_component(function_search_names=['Pylon_08'], name='pylon8')
pylon9 = geometry.declare_component(function_search_names=['Pylon_09'], name='pylon9')
pylon10 = geometry.declare_component(function_search_names=['Pylon_10'], name='pylon10')
pylon11 = geometry.declare_component(function_search_names=['Pylon_11'], name='pylon11')
pylon12 = geometry.declare_component(function_search_names=['Pylon_12'], name='pylon12')

nacelle7 = geometry.declare_component(function_search_names=['HLNacelle_7_Tail'], name='nacelle7')
nacelle8 = geometry.declare_component(function_search_names=['HLNacelle_8_Tail'], name='nacelle8')
nacelle9 = geometry.declare_component(function_search_names=['HLNacelle_9_Tail'], name='nacelle9')
nacelle10 = geometry.declare_component(function_search_names=['HLNacelle_10_Tail'], name='nacelle10')
nacelle11 = geometry.declare_component(function_search_names=['HLNacelle_11_Tail'], name='nacelle11')
nacelle12 = geometry.declare_component(function_search_names=['HLNacelle_12_Tail'], name='nacelle12')

spinner1 = geometry.declare_component(function_search_names=['HL_Spinner12, 0'], name='spinner1')
spinner2 = geometry.declare_component(function_search_names=['HL_Spinner11, 0'], name='spinner2')
spinner3 = geometry.declare_component(function_search_names=['HL_Spinner10, 0'], name='spinner3')
spinner4 = geometry.declare_component(function_search_names=['HL_Spinner9, 0'], name='spinner4')
spinner5 = geometry.declare_component(function_search_names=['HL_Spinner8, 0'], name='spinner5')
spinner6 = geometry.declare_component(function_search_names=['HL_Spinner7, 0'], name='spinner6')
spinner7 = geometry.declare_component(function_search_names=['HL_Spinner7, 1'], name='spinner7')
spinner8 = geometry.declare_component(function_search_names=['HL_Spinner8, 1'], name='spinner8')
spinner9 = geometry.declare_component(function_search_names=['HL_Spinner9, 1'], name='spinner9')
spinner10 = geometry.declare_component(function_search_names=['HL_Spinner10, 1'], name='spinner10')
spinner11 = geometry.declare_component(function_search_names=['HL_Spinner11, 1'], name='spinner11')
spinner12 = geometry.declare_component(function_search_names=['HL_Spinner12, 1'], name='spinner12')


prop = geometry.declare_component(function_search_names=['HL-Prop'], name='prop')
motor = geometry.declare_component(function_search_names=['HL_Motor'], name='motor')
motor_interface = geometry.declare_component(function_search_names=['HL_Motor_Controller_Interface'], name='motor_interface')

cruise_spinner1 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 0'], name='cruise_spinner1')
cruise_spinner2 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 1'], name='cruise_spinner2')

cruise_motor =  geometry.declare_component(function_search_names=['CruiseNacelle-Motor'], name='cruise_motor')
cruise_nacelle =  geometry.declare_component(function_search_names=['CruiseNacelle-Tail'], name='cruise_nacelle')
cruise_prop = geometry.declare_component(function_search_names=['Cruise-Prop'], name='cruise_prop')

M1_components = [pylon7, nacelle7, spinner1, prop, motor, motor_interface]
M2_components = [pylon8, nacelle8, spinner2, prop, motor, motor_interface]
M3_components = [pylon9, nacelle9, spinner3, prop, motor, motor_interface]
M4_components = [pylon10, nacelle10, spinner4, prop, motor, motor_interface]
M5_components = [pylon11, nacelle11, spinner5, prop, motor, motor_interface]
M6_components = [pylon12, nacelle12, spinner6, prop, motor, motor_interface]
M7_components = [pylon7, nacelle7, spinner7, prop, motor, motor_interface]
M8_components = [pylon8, nacelle8, spinner8, prop, motor, motor_interface]
M9_components = [pylon9, nacelle9, spinner9, prop, motor, motor_interface]
M10_components = [pylon10, nacelle10, spinner10, prop, motor, motor_interface]
M11_components = [pylon11, nacelle11, spinner11, prop, motor, motor_interface]
M12_components = [pylon12, nacelle12, spinner12, prop, motor, motor_interface]
CM1_components = [cruise_nacelle, cruise_spinner1, cruise_prop, cruise_motor]
CM2_components = [cruise_nacelle, cruise_spinner2, cruise_prop, cruise_motor]
total_HL_motor_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components]
total_prop_sys_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components, CM1_components, CM2_components]


# geometry.plot()

# Wing Region Info
wing_le_left_guess = np.array([-12.356, -16, -5.5])*ft2m
wing_le_left_parametric = wing.project(wing_le_left_guess, plot=False)
wing_le_left = geometry.evaluate(wing_le_left_parametric)

wing_le_right_guess = np.array([-12.356, 16, -5.5])*ft2m
wing_le_right_parametric = wing.project(wing_le_right_guess, plot=False)
wing_le_right = geometry.evaluate(wing_le_right_parametric)

wing_le_center_guess = np.array([-12.356, 0., -5.5])*ft2m
wing_le_center_parametric = wing.project(wing_le_center_guess, plot=False)
wing_le_center = geometry.evaluate(wing_le_center_parametric)

wing_te_left_guess = np.array([-14.25, -16, -5.5])*ft2m
wing_te_left_parametric = wing.project(wing_te_left_guess, plot=False)
wing_te_left = geometry.evaluate(wing_te_left_parametric)

wing_te_right_guess = np.array([-14.25, 16, -5.5])*ft2m
wing_te_right_parametric = wing.project(wing_te_right_guess, plot=False)
wing_te_right = geometry.evaluate(wing_te_right_parametric)

wing_te_center_guess = np.array([-14.25, 0., -5.5])*ft2m
wing_te_center_parametric = wing.project(wing_te_center_guess, plot=False)
wing_te_center = geometry.evaluate(wing_te_center_parametric)

wing_qc_center_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False)
wing_qc_tip_right_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), 16., -5.5])*ft2m, plot=False)
wing_qc_tip_left_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), -16., -5.5])*ft2m, plot=False)


wing_parametric_geometry = [
    wing_le_left_parametric,
    wing_le_right_parametric,
    wing_le_center_parametric,
    wing_te_left_parametric,
    wing_te_right_parametric,
    wing_te_center_parametric,
    wing_qc_center_parametric,
    wing_qc_tip_right_parametric,
    wing_qc_tip_left_parametric
]

wingspan = np.linalg.norm(wing_le_left.value - wing_le_right.value)
# print('Wingspan: ', wingspan)

## ADD CONTROL SURFACE INFO PROJECTIONS HERE

left_aileron_le_left_guess = np.array([-13.85, -15.5, -7.5])*ft2m
left_aileron_le_left_parametric = aileronL.project(left_aileron_le_left_guess, plot=False)
left_aileron_le_left = geometry.evaluate(left_aileron_le_left_parametric)

left_aileron_le_right_guess = np.array([-13.85, -11.4, -7.5])*ft2m
left_aileron_le_right_parametric = aileronL.project(left_aileron_le_right_guess, plot=False)
left_aileron_le_right = geometry.evaluate(left_aileron_le_right_parametric)

left_aileron_le_center_guess = np.array([-13.85, -13.4, -7.5])*ft2m
left_aileron_le_center_parametric = aileronL.project(left_aileron_le_center_guess, plot=False)
left_aileron_le_center = geometry.evaluate(left_aileron_le_center_parametric)

left_aileron_le_center_on_wing_te_guess = left_aileron_le_center_guess
left_aileron_le_center_on_wing_te_parametric = wing.project(left_aileron_le_center_on_wing_te_guess, plot=False)

left_aileron_te_center_parametric = aileronL.project(np.array([-14.25, -13.4, -7.5])*ft2m, plot=False)
left_aileron_te_center = geometry.evaluate(left_aileron_te_center_parametric)

left_aileron_te_left_parametric = aileronL.project(np.array([-14.25, -15.5, -7.5])*ft2m, plot=False)
left_aileron_te_left = geometry.evaluate(left_aileron_te_left_parametric)

left_aileron_te_right_parametric = aileronL.project(np.array([-14.25, -11.4, -7.5])*ft2m, plot=False)
left_aileron_te_right = geometry.evaluate(left_aileron_te_right_parametric)

left_aileron_qc_center_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -13.4, -7.5])*ft2m, plot=False)
left_aileron_qc_tip_right_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.4, -7.5])*ft2m, plot=False)
left_aileron_qc_tip_left_parametric = aileronL.project(np.array([-13+(0.25*(-14.25+13)), -15.15, -7.5])*ft2m, plot=False)

left_aileron_span = np.linalg.norm(left_aileron_le_left.value - left_aileron_le_right.value)
# print('Left Aileron Span: ', left_aileron_span)

left_aileron_chord = np.linalg.norm(left_aileron_le_center[0].value - left_aileron_te_center[0].value)
# print('Left Aileron Chord: ', left_aileron_chord)

left_aileron_parametric_geometry = [
    left_aileron_le_left_parametric,
    left_aileron_le_right_parametric,
    left_aileron_le_center_parametric,
    left_aileron_te_left_parametric,
    left_aileron_te_right_parametric,
    left_aileron_te_center_parametric,
    left_aileron_qc_center_parametric,
    left_aileron_qc_tip_right_parametric,
    left_aileron_qc_tip_left_parametric
]

right_aileron_le_left_guess = np.array([-13.85, 11.4, -7.5])*ft2m
right_aileron_le_left_parametric = aileronR.project(right_aileron_le_left_guess, plot=False)
right_aileron_le_left = geometry.evaluate(right_aileron_le_left_parametric)

right_aileron_le_right_guess = np.array([-13.85, 15.5, -7.5])*ft2m
right_aileron_le_right_parametric = aileronR.project(right_aileron_le_right_guess, plot=False)
right_aileron_le_right = geometry.evaluate(right_aileron_le_right_parametric)

right_aileron_le_center_guess = np.array([-13.85, 13.4, -7.5])*ft2m
right_aileron_le_center_parametric = aileronR.project(right_aileron_le_center_guess, plot=False)
right_aileron_le_center = geometry.evaluate(right_aileron_le_center_parametric)

right_aileron_le_center_on_wing_te_guess = right_aileron_le_center_guess
right_aileron_le_center_on_wing_te_parametric = wing.project(right_aileron_le_center_on_wing_te_guess, plot=False)

right_aileron_te_center_parametric = aileronR.project(np.array([-14.25, 13.4, -7.5])*ft2m, plot=False)
right_aileron_te_center = geometry.evaluate(right_aileron_te_center_parametric)

right_aileron_te_left_parametric = aileronR.project(np.array([-14.25, 11.4, -7.5])*ft2m, plot=False)
right_aileron_te_left = geometry.evaluate(right_aileron_te_left_parametric)

right_aileron_te_right_parametric = aileronR.project(np.array([-14.25, 15.5, -7.5])*ft2m, plot=False)
right_aileron_te_right = geometry.evaluate(right_aileron_te_right_parametric)

right_aileron_qc_center_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 13.12, -7.5])*ft2m, plot=False)
right_aileron_qc_tip_right_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 11.4, -7.5])*ft2m, plot=False)
right_aileron_qc_tip_left_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 15.15, -7.5])*ft2m, plot=False)
                                                         
right_aileron_parametric_geometry = [
    right_aileron_le_left_parametric,
    right_aileron_le_right_parametric,
    right_aileron_le_center_parametric,
    right_aileron_te_left_parametric,
    right_aileron_te_right_parametric,
    right_aileron_te_center_parametric,
    right_aileron_qc_center_parametric,
    right_aileron_qc_tip_right_parametric,
    right_aileron_qc_tip_left_parametric
]


right_aileron_span = np.linalg.norm(right_aileron_le_left.value - right_aileron_le_right.value)
# print('Right Aileron Span: ', right_aileron_span)

right_aileron_chord = np.linalg.norm(right_aileron_le_center[0].value - right_aileron_te_center[0].value)
# print('Right Aileron Chord: ', right_aileron_chord)

left_flap_le_left_guess = np.array([-13.85, -11.5, -7.5])*ft2m
left_flap_le_left_parametric = flapL.project(left_flap_le_left_guess, plot=False)
left_flap_le_left = geometry.evaluate(left_flap_le_left_parametric)

left_flap_le_right_guess = np.array([-13.85, -0.6, -7.5])*ft2m
left_flap_le_right_parametric = flapL.project(left_flap_le_right_guess, plot=False)
left_flap_le_right = geometry.evaluate(left_flap_le_right_parametric)

left_flap_le_center_guess = np.array([-13.85, -6.05, -7.5])*ft2m
left_flap_le_center_parametric = flapL.project(left_flap_le_center_guess, plot=False)
left_flap_le_center = geometry.evaluate(left_flap_le_center_parametric)

left_flap_le_center_on_wing_te_guess = left_flap_le_center_guess
left_flap_le_center_on_wing_te_parametric = wing.project(left_flap_le_center_on_wing_te_guess, plot=False)

left_flap_te_center_parametric = flapL.project(np.array([-14.25, -6.05, -5.5])*ft2m, plot=False)
left_flap_te_center = geometry.evaluate(left_flap_te_center_parametric)

left_flap_te_left_parametric = flapL.project(np.array([-14.25, -11.5, -5.5])*ft2m, plot=False)
left_flap_te_left = geometry.evaluate(left_flap_te_left_parametric)

left_flap_te_right_parametric = flapL.project(np.array([-14.25, -0.6, -5.5])*ft2m, plot=False)
left_flap_te_right = geometry.evaluate(left_flap_te_right_parametric)

left_flap_qc_center_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -6.56, -5.5])*ft2m, plot=False)
left_flap_qc_tip_right_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -0.6, -5.5])*ft2m, plot=False)
left_flap_qc_tip_left_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.48, -5.5])*ft2m, plot=False)
                                                  
left_flap_parametric_geometry = [
    left_flap_le_left_parametric,
    left_flap_le_right_parametric,
    left_flap_le_center_parametric,
    left_flap_te_left_parametric,
    left_flap_te_right_parametric,
    left_flap_te_center_parametric,
    left_flap_qc_center_parametric,
    left_flap_qc_tip_right_parametric,
    left_flap_qc_tip_left_parametric
]

left_flap_span = np.linalg.norm(left_flap_le_left.value - left_flap_le_right.value)
# print('Left Flap Span: ', left_flap_span)
left_flap_chord = np.linalg.norm(left_flap_le_center[0].value - left_flap_te_center[0].value)
# print('Left Flap Chord: ', left_flap_chord)


right_flap_le_left_guess = np.array([-13.85, 0.6, -7.5])*ft2m
right_flap_le_left_parametric = flapR.project(right_flap_le_left_guess, plot=False)
right_flap_le_left = geometry.evaluate(right_flap_le_left_parametric)

right_flap_le_right_guess = np.array([-13.85, 11.5, -7.5])*ft2m
right_flap_le_right_parametric = flapR.project(right_flap_le_right_guess, plot=False)
right_flap_le_right = geometry.evaluate(right_flap_le_right_parametric)

right_flap_le_center_guess = np.array([-13.85, 6.05, -7.5])*ft2m
right_flap_le_center_parametric = flapR.project(right_flap_le_center_guess, plot=False)
right_flap_le_center = geometry.evaluate(right_flap_le_center_parametric)

right_flap_le_center_on_wing_te_guess = right_flap_le_center_guess
right_flap_le_center_on_wing_te_parametric = wing.project(right_flap_le_center_on_wing_te_guess, plot=False)

right_flap_te_center_parametric = flapR.project(np.array([-14.25, 6.05, -7.5])*ft2m, plot=False)
right_flap_te_center = geometry.evaluate(right_flap_te_center_parametric)

right_flap_te_left_parametric = flapR.project(np.array([-14.25, 0.6, -7.5])*ft2m, plot=False)
right_flap_te_left = geometry.evaluate(right_flap_te_left_parametric)

right_flap_te_right_parametric = flapR.project(np.array([-14.25, 11.5, -7.5])*ft2m, plot=False)
right_flap_te_right = geometry.evaluate(right_flap_te_right_parametric)

right_flap_qc_center_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 6.56, -7.5])*ft2m, plot=False)
right_flap_qc_tip_right_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 11.48, -7.5])*ft2m, plot=False)
right_flap_qc_tip_left_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 1.5, -7.5])*ft2m, plot=False)

right_flap_parametric_geometry = [
    right_flap_le_left_parametric,
    right_flap_le_right_parametric,
    right_flap_le_center_parametric,
    right_flap_te_left_parametric,
    right_flap_te_right_parametric,
    right_flap_te_center_parametric,
    right_flap_qc_center_parametric,
    right_flap_qc_tip_right_parametric,
    right_flap_qc_tip_left_parametric
]

right_flap_span = np.linalg.norm(right_flap_le_left.value - right_flap_le_right.value)
# print('Right Flap Span: ', right_flap_span)
right_flap_chord = np.linalg.norm(right_flap_le_center[0].value - right_flap_te_center[0].value)
# print('Right Flap Chord: ', right_flap_chord)

# HT Region Info
ht_le_left_parametric = h_tail.project(np.array([-26.5, -5.25, -5.5])*ft2m, plot=False)
ht_le_left = geometry.evaluate(ht_le_left_parametric)

ht_le_center_parametric = h_tail.project(np.array([-27, 0., -5.5])*ft2m, plot=False)
ht_le_center = geometry.evaluate(ht_le_center_parametric)

ht_le_right_parametric = h_tail.project(np.array([-26.5, 5.25, -5.5])*ft2m, plot=False)
ht_le_right = geometry.evaluate(ht_le_right_parametric)

ht_te_left_parametric = h_tail.project(np.array([-30, -5.25, -5.5])*ft2m, plot=False)
ht_te_left = geometry.evaluate(ht_te_left_parametric)

ht_te_center_guess = np.array([-30, 0., -5.5])*ft2m
ht_te_center_parametric = h_tail.project(ht_te_center_guess, plot=False)
ht_te_center = geometry.evaluate(h_tail.project(ht_te_center_guess, plot=False))

ht_te_right_parametric = h_tail.project(np.array([-30, 5.25, -5.5])*ft2m, plot=False)
ht_te_right = geometry.evaluate(ht_te_right_parametric)

ht_qc_center_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False)
ht_qc_tip_right_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), 5.25, -5.5])*ft2m, plot=False)
ht_qc_tip_left_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), -5.25, -5.5])*ft2m, plot=False)
                                            
ht_qc = geometry.evaluate(h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))

ht_parametric_geometry = [
    ht_le_left_parametric,
    ht_le_right_parametric,
    ht_le_center_parametric,
    ht_te_left_parametric,
    ht_te_right_parametric,
    ht_te_center_parametric,
    ht_qc_center_parametric,
    ht_qc_tip_right_parametric,
    ht_qc_tip_left_parametric
]


ht_span = np.linalg.norm(ht_le_left.value - ht_le_right.value)
# print('Horizontal Tail Span: ', ht_span)
ht_chord = np.linalg.norm(ht_le_center[0].value - ht_te_center[0].value)
# print('Horizontal Tail Chord: ', ht_chord)

trimTab_le_left_parametric = trimTab.project(np.array([-29.4, -3.3, -5.5])*ft2m, plot=False)
trimTab_le_right_parametric = trimTab.project(np.array([-29.4, 3.3, -5.5])*ft2m, plot=False)
trimTab_le_left = geometry.evaluate(trimTab_le_left_parametric)
trimTab_le_center_parametric = trimTab.project(np.array([-29.4, 0, -5.5])*ft2m, plot=False)
trimTab_le_center = geometry.evaluate(trimTab_le_center_parametric)
trimTab_le_right = geometry.evaluate(trimTab_le_right_parametric)

trimTab_te_left_parametric = trimTab.project(np.array([-30, -3.3, -5.5])*ft2m, plot=False)
trimTab_te_center_parametric = trimTab.project(np.array([-30, 0, -5.5])*ft2m, plot=False)
trimTab_te_right_parametric = trimTab.project(np.array([-30, 3.3, -5.5])*ft2m, plot=False)

trimTab_te_center = geometry.evaluate(trimTab.project(np.array([-30, 0, -5.5])*ft2m, plot=False))

trimTab_qc_center_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), 0, -5.5])*ft2m, plot=False)
trimTab_qc_tip_right_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), 3.3, -5.5])*ft2m, plot=False)
trimTab_qc_tip_left_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), -3.3, -5.5])*ft2m, plot=False)                                                  


trimTab_parametric_geometry = [
    trimTab_le_left_parametric,
    trimTab_le_right_parametric,
    trimTab_le_center_parametric,
    trimTab_te_left_parametric,
    trimTab_te_right_parametric,
    trimTab_te_center_parametric,
    trimTab_qc_center_parametric,
    trimTab_qc_tip_right_parametric,
    trimTab_qc_tip_left_parametric
]


trim_tab_span = np.linalg.norm(trimTab_le_left.value - trimTab_le_right.value)
# print('Trim Tab Span: ', trim_tab_span)

trimTab_chord = np.linalg.norm(trimTab_le_center[0].value - trimTab_te_center[0].value)
# print('Trim Tab Chord: ', trimTab_chord)

# VT Region Info
vt_le_base_parametric = vertTail.project(np.array([-23, 0, -5.5])*ft2m, plot=False)
vt_le_base = geometry.evaluate(vt_le_base_parametric)
vt_le_mid_parametric = vertTail.project(np.array([-26, 0., -8])*ft2m, plot=False)
vt_le_mid = geometry.evaluate(vt_le_mid_parametric)
vt_le_tip_parametric = vertTail.project(np.array([-28.7, 0, -11])*ft2m, plot=False)
vt_le_tip = geometry.evaluate(vt_le_tip_parametric)

vt_te_base_parametric = vertTail.project(np.array([-27.75, 0, -5.5])*ft2m, plot=False)
vt_te_base = geometry.evaluate(vt_te_base_parametric)
vt_te_mid_parametric = vertTail.project(np.array([-28.7, 0., -8])*ft2m, plot=False)
vt_te_mid= geometry.evaluate(vt_te_mid_parametric)
vt_te_tip_parametric = vertTail.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False)
vt_te_tip = geometry.evaluate(vt_te_tip_parametric)
vt_qc = geometry.evaluate(vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m, plot=False))
vt_qc_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)
vt_qc_base_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)
vt_qc_tip_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)


vt_parametric_geometry = [
    vt_le_base_parametric,
    vt_le_tip_parametric,
    vt_le_base_parametric,
    vt_te_base_parametric,
    vt_te_tip_parametric,
    vt_te_mid_parametric,
    vt_qc_parametric,
    vt_qc_base_parametric,
    vt_qc_tip_parametric
]

vt_span = np.linalg.norm(vt_te_base.value - vt_te_tip.value)
# print('Vertical Tail Span: ', vt_span)

vt_chord = np.linalg.norm(vt_le_mid.value - vt_te_mid.value)
# print('Vertical Tail Chord: ', vt_chord)

rudder_le_base = geometry.evaluate(rudder.project(np.array([-23, 0, -5.5])*ft2m, plot=False))
rudder_le_mid_parametric = rudder.project(np.array([-28.7, 0., -8.])*ft2m, plot=False)
rudder_le_mid = geometry.evaluate(rudder_le_mid_parametric)
rudder_le_tip = geometry.evaluate(rudder.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False))
rudder_span = np.linalg.norm(rudder_le_base.value - rudder_le_tip.value)
# print('Rudder Span: ', rudder_span)
rudder_chord = np.linalg.norm(rudder_le_mid.value - rudder_le_tip.value)
# print('Rudder Chord: ', rudder_chord)



# Fuselage Region Info
fuselage_wing_qc = geometry.evaluate(fuselage.project(np.array([-12.356+(0.25*(-14.25+12.356))*ft2m, 0., -5.5]), plot=False))
fuselage_wing_te_center_parametric = fuselage.project(np.array([-14.25, 0., -5.5])*ft2m, plot=False)
fuselage_wing_te_center = geometry.evaluate(fuselage_wing_te_center_parametric)
fuselage_tail_qc = geometry.evaluate(fuselage.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))
fuselage_tail_te_center = geometry.evaluate(fuselage.project(np.array([-30, 0., -5.5])*ft2m, plot=False))


# Propeller Region Info
M1_disk_pt =  np.array([-12.5, 14, -7.355])*ft2m
M12_disk_pt = np.array([-12.5, -14, -7.355])*ft2m
M2_disk_pt =  np.array([-12.35, 12, -7.355])*ft2m
M11_disk_pt = np.array([-12.35, -12, -7.355])*ft2m
M3_disk_pt = np.array([-12.2, 10, -7.659])*ft2m
M10_disk_pt = np.array([-12.2, -10, -7.659])*ft2m
M4_disk_pt = np.array([-12, 8, -7.659])*ft2m
M9_disk_pt = np.array([-12, -8, -7.659])*ft2m
M5_disk_pt = np.array([-11.8, 6, -7.659])*ft2m
M8_disk_pt = np.array([-11.8, -6, -7.659])*ft2m
M6_disk_pt = np.array([-11.6, 4, -7.659])*ft2m
M7_disk_pt = np.array([-11.6, -4, -7.659])*ft2m

M1_disk_on_wing = spinner1.project(M1_disk_pt, plot=False)
M1_disk = geometry.evaluate(M1_disk_on_wing)
# print('From aircraft, Left Outermost Disk (ft): ', M1_disk.value)

M2_disk_on_wing = spinner2.project(M2_disk_pt, plot=False)
M2_disk = geometry.evaluate(M2_disk_on_wing)
# print('From aircraft, Left Outer Disk (ft): ', M2_disk.value)

M3_disk_on_wing = spinner3.project(M3_disk_pt, plot=False)
M3_disk = geometry.evaluate(M3_disk_on_wing)
# print('From aircraft, Left Inner Disk (ft): ', M3_disk.value)

M4_disk_on_wing = spinner4.project(M4_disk_pt, plot=False)
M4_disk = geometry.evaluate(M4_disk_on_wing)
# print('From aircraft, Left Innermost Disk (ft): ', M4_disk.value)

M5_disk_on_wing = spinner5.project(M5_disk_pt, plot=False)
M5_disk = geometry.evaluate(M5_disk_on_wing)
# print('From aircraft, Left Disk (ft): ', M5_disk.value)

M6_disk_on_wing = spinner6.project(M6_disk_pt, plot=False)
M6_disk = geometry.evaluate(M6_disk_on_wing)
# print('From aircraft, Right Disk (ft): ', M6_disk.value)

M7_disk_on_wing = spinner7.project(M7_disk_pt, plot=False)
M7_disk = geometry.evaluate(M7_disk_on_wing)
# print('From aircraft, Right Disk (ft): ', M7_disk.value)

M8_disk_on_wing = spinner8.project(M8_disk_pt, plot=False)
M8_disk = geometry.evaluate(M8_disk_on_wing)
# print('From aircraft, Right Disk (ft): ', M8_disk.value)

M9_disk_on_wing = spinner9.project(M9_disk_pt, plot=False)
M9_disk = geometry.evaluate(M9_disk_on_wing)
# print('From aircraft, Right Innermost Disk (ft): ', M9_disk.value)

M10_disk_on_wing = spinner10.project(M10_disk_pt, plot=False)
M10_disk = geometry.evaluate(M10_disk_on_wing)
# print('From aircraft, Right Inner Disk (ft): ', M10_disk.value)

M11_disk_on_wing = spinner11.project(M11_disk_pt, plot=False)
M11_disk = geometry.evaluate(M11_disk_on_wing)
# print('From aircraft, Right Outer Disk (ft): ', M11_disk.value)

M12_disk_on_wing = spinner12.project(M12_disk_pt, plot=False)
M12_disk = geometry.evaluate(M12_disk_on_wing)
# print('From aircraft, Right Outermost Disk (ft): ', M12_disk.value)

MotorDisks = [M1_disk, M2_disk, M3_disk, M4_disk, M5_disk, M6_disk, M7_disk, M8_disk, M9_disk, M10_disk, M11_disk, M12_disk]


fuselage_nose_guess = np.array([-1.75, 0, -4])*ft2m
fuselage_rear_guess = np.array([-29.5, 0, -5.5])*ft2m
fuselage_nose_pts_parametric = fuselage.project(fuselage_nose_guess, grid_search_density_parameter=20, plot=False)
fuselage_nose = geometry.evaluate(fuselage_nose_pts_parametric)
fuselage_rear_pts_parametric = fuselage.project(fuselage_rear_guess, plot=False)
fuselage_rear = geometry.evaluate(fuselage_rear_pts_parametric)

# For Cruise Motor 1 Hub Region
cruise_motor1_tip_guess = np.array([-13, -15.83, -5.5])*ft2m
cruise_motor1_tip_parametric = cruise_spinner1.project(cruise_motor1_tip_guess, plot=False)
cruise_motor1_tip = geometry.evaluate(cruise_motor1_tip_parametric)
# print('From aircraft, cruise motor hub tip (ft): ', cruise_motor_tip.value)

cruise_motor1_base_guess = cruise_motor1_tip + np.array([-1.67, 0, 0])*ft2m
cruise_motor1_base_parametric = cruise_spinner1.project(cruise_motor1_base_guess, plot=False)
cruise_motor1_base= geometry.evaluate(cruise_motor1_base_parametric)
# print('From aircraft, cruise motor hub base (ft): ', cruise_motor_base.value)

# For Cruise Motor 2 Hub Region
cruise_motor2_tip_guess = np.array([-13, 15.83, -5.5])*ft2m
cruise_motor2_tip_parametric = cruise_spinner2.project(cruise_motor2_tip_guess, plot=False)
cruise_motor2_tip = geometry.evaluate(cruise_motor1_tip_parametric)
# print('From aircraft, cruise motor hub tip (ft): ', cruise_motor_tip.value)

cruise_motor2_base_guess = cruise_motor2_tip + np.array([-1.67, 0, 0])*ft2m
cruise_motor2_base_parametric = cruise_spinner2.project(cruise_motor2_base_guess, plot=False)
cruise_motor2_base= geometry.evaluate(cruise_motor2_base_parametric)
# print('From aircraft, cruise motor hub base (ft): ', cruise_motor_base.value)

wing_on_cruise_motor1_parametric = cruise_spinner2.project(wing_le_left_guess, plot=False)
wing_on_cruise_motor2_parametric = cruise_spinner1.project(wing_le_right_guess, plot=False)





## AXIS/AXISLSDOGEO CREATION


def axes_create():

    # OpenVSP Model Axis
    openvsp_axis = Axis(
        name='OpenVSP Axis',
        x = np.array([0, ]) * ureg.foot,
        y = np.array([0, ])* ureg.foot,
        z = np.array([0, ])* ureg.foot,
        origin=ValidOrigins.OpenVSP.value
    )
    
    wing_axis = AxisLsdoGeo(
    name='Wing Axis',
    geometry=wing,
    parametric_coords=wing_le_center_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
    )
    # print('Wing axis translation (ft): ', wing_axis.translation.value)
    # print('Wing axis rotation (deg): ', np.rad2deg(wing_axis.euler_angles_vector.value))
    # geometry.plot()

    left_flap_axis = AxisLsdoGeo(
    name='Left Flap Axis',
    geometry=flapL,
    parametric_coords=left_flap_le_left_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
    )

    right_flap_axis = AxisLsdoGeo(
    name='Right Flap Axis',
    geometry=flapR,
    parametric_coords=right_flap_le_left_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
    )

    left_aileron_axis = AxisLsdoGeo(
    name='Left Aileron Axis',
    geometry=aileronL,
    parametric_coords=left_aileron_le_left_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
    )

    right_aileron_axis = AxisLsdoGeo(
    name='Right Aileron Axis',
    geometry=aileronR,
    parametric_coords=right_aileron_le_left_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
    )

    ## Tail Region Axis


    # ht_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='HT incidence')
    # trimTab_deflection = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='Trim Tab Deflection')

    # h_tail.rotate(ht_le_center, np.array([0., 1., 0.]), angles=ht_incidence)
    # trimTab.rotate(ht_le_center, np.array([0., 1., 0.]), angles=ht_incidence)

    ht_tail_axis = AxisLsdoGeo(
        name='Horizontal Tail Axis',
        geometry=h_tail,
        parametric_coords=ht_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi = np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('HT axis translation (ft): ', ht_tail_axis.translation.value)
    # print('HT axis rotation (deg): ', np.rad2deg(ht_tail_axis.euler_angles_vector.value))

    trimTab_axis = AxisLsdoGeo(
        name='Trim Tab Axis',
        geometry=trimTab,
        parametric_coords=trimTab_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi = np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=ht_tail_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    # trimTab.rotate(trimTab_le_center, np.array([0.,1.,0.]),angles=trimTab_deflection)


    # print('Trim Tab axis translation (ft): ', trimTab_axis.translation.value)
    # print('Trim Tab axis rotation (deg): ', np.rad2deg(trimTab_axis.euler_angles_vector.value))
    # geometry.plot()



    # rudder_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='VT incidence')
    # rudder.rotate(rudder_le_mid, np.array([0., 0., 1.]), angles=rudder_incidence)


    vt_tail_axis = AxisLsdoGeo(
        name='Vertical Tail Axis',
        geometry=rudder,
        parametric_coords=rudder_le_mid_parametric,
        sequence=np.array([3, 2, 1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('VT axis translation (ft): ', vt_tail_axis.translation.value)
    # print('VT axis rotation (deg): ', np.rad2deg(vt_tail_axis.euler_angles_vector.value))
    # geometry.plot()

    ## Distributed Propulsion Motors Axes

    M1_axis = AxisLsdoGeo(
        name= 'Motor 1 Axis',
        geometry=spinner1,
        parametric_coords=M1_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Left Outermost motor axis translation (ft): ', l_om_axis.translation.value)
    # print('Left Outermost motor axis rotation (deg): ', np.rad2deg(l_om_axis.euler_angles_vector.value))

    M2_axis = AxisLsdoGeo(
        name= 'Motor 2 Axis',
        geometry=spinner2,
        parametric_coords=M2_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Left Outer motor axis translation (ft): ', l_o_axis.translation.value)
    # print('Left Outer motor axis rotation (deg): ', np.rad2deg(l_o_axis.euler_angles_vector.value))

    M3_axis = AxisLsdoGeo(
        name= 'Motor 3 Axis',
        geometry=spinner3,
        parametric_coords=M3_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M4_axis = AxisLsdoGeo(
        name= 'Motor 4 Axis',
        geometry=spinner4,
        parametric_coords=M4_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M5_axis = AxisLsdoGeo(
        name= 'Motor 5 Axis',
        geometry=spinner5,
        parametric_coords=M5_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M6_axis = AxisLsdoGeo(
        name= 'Motor 6 Axis',
        geometry=spinner6,
        parametric_coords=M6_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M7_axis = AxisLsdoGeo(
        name= 'Motor 7 Axis',
        geometry=spinner7,
        parametric_coords=M7_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M8_axis = AxisLsdoGeo(
        name= 'Motor 8 Axis',
        geometry=spinner8,
        parametric_coords=M8_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M9_axis = AxisLsdoGeo(
        name= 'Motor 9 Axis',
        geometry=spinner9,
        parametric_coords=M9_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M10_axis = AxisLsdoGeo(
        name= 'Motor 10 Axis',
        geometry=spinner10,
        parametric_coords=M10_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M11_axis = AxisLsdoGeo(
        name= 'Motor 11 Axis',
        geometry=spinner11,
        parametric_coords=M11_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M12_axis = AxisLsdoGeo(
        name= 'Motor 12 Axis',
        geometry=spinner12,
        parametric_coords=M12_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Right Outermost motor axis translation (ft): ', r_om_axis.translation.value)
    # print('Right Outermost motor axis rotation (deg): ', np.rad2deg(r_om_axis.euler_angles_vector.value))

    HL_motor_axes = [M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis]
    # Cruise Motor Region

    @dataclass
    class CruiseMotorRotation(csdl.VariableGroup):
        cant : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree
        pitch : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(0), name='CruiseMotorPitchAngle')
        yaw : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree
    cruise_spinner_rotation = CruiseMotorRotation()
    # cruise_spinner.rotate(cruise_motor_base, np.array([0., 1., 0.]), angles=cruise_spinner_rotation.pitch)

    cruise_motor1_axis = AxisLsdoGeo(
        name= 'Cruise Motor 1 Axis',
        geometry=cruise_spinner1,
        parametric_coords=cruise_motor1_tip_parametric,
        sequence=np.array([3,2,1]),
        phi=cruise_spinner_rotation.cant,
        theta=cruise_spinner_rotation.pitch,
        psi=cruise_spinner_rotation.yaw,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Cruise motor axis translation (ft): ', cruise_motor_axis.translation.value)
    # print('Cruise motor axis rotation (deg): ', np.rad2deg(cruise_motor_axis.euler_angles_vector.value))

    cruise_motor2_axis = AxisLsdoGeo(
        name= 'Cruise Motor 2 Axis',
        geometry=cruise_spinner2,
        parametric_coords=cruise_motor2_tip_parametric,
        sequence=np.array([3,2,1]),
        phi=cruise_spinner_rotation.cant,
        theta=cruise_spinner_rotation.pitch,
        psi=cruise_spinner_rotation.yaw,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Cruise motor axis translation (ft): ', cruise_motor_axis.translation.value)
    # print('Cruise motor axis rotation (deg): ', np.rad2deg(cruise_motor_axis.euler_angles_vector.value))

    cruise_motor_axes = [cruise_motor1_axis, cruise_motor2_axis]

    inertial_axis = Axis(
        name='Inertial Axis',
        x=np.array([0, ]) * ureg.meter,
        y=np.array([0, ]) * ureg.meter,
        z=np.array([0, ]) * ureg.meter,
        origin=ValidOrigins.Inertial.value
    )

    fd_axis = Axis(
        name='Flight Dynamics Body Fixed Axis',
        x = np.array([0, ]) * ureg.meter,
        y = np.array([0, ]) * ureg.meter,
        z = np.array([0, ]) * ureg.meter,
        phi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='phi'),
        theta=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='theta'),
        psi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='psi'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )
    # print('Body-fixed angles (deg)', np.rad2deg(fd_axis.euler_angles_vector.value))



    # Aircraft Wind Axis
    @dataclass
    class WindAxisRotations(csdl.VariableGroup):
        mu : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree # bank
        gamma : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(2), name='Flight path angle')
        xi : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree  # Heading
    wind_axis_rotations = WindAxisRotations()

    wind_axis = Axis(
        name='Wind Axis',
        x = np.array([0, ]) * ureg.meter,
        y = np.array([0, ]) * ureg.meter,
        z = np.array([0, ]) * ureg.meter,
        phi=wind_axis_rotations.mu,
        theta=wind_axis_rotations.gamma,
        psi=wind_axis_rotations.xi,
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )
    # print('Wind axis angles (deg)', np.rad2deg(wind_axis.euler_angles_vector.value))

    return openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geometry, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis

openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis,geometry,left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis = axes_create()




velocity_vector_in_wind = Vector(vector=csdl.Variable(shape=(3,), value=np.array([-1, 0, 0]), name='wind_vector'), axis=wind_axis)
# print('Unit wind vector in wind axis: ', velocity_vector_in_wind.vector.value)

R_wind_to_inertial = build_rotation_matrix(wind_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_inertial =  Vector(csdl.matvec(R_wind_to_inertial, velocity_vector_in_wind.vector), axis=inertial_axis)
# print('Unit wind vector in inertial axis: ', wind_vector_in_inertial.vector.value)

R_body_to_inertial = build_rotation_matrix(fd_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_body =  Vector(csdl.matvec(csdl.transpose(R_body_to_inertial), wind_vector_in_inertial.vector), axis=fd_axis)
# print('Unit wind vector in body axis: ', wind_vector_in_body.vector.value)

R_wing_to_openvsp = build_rotation_matrix(wing_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_wing =  Vector(csdl.matvec(csdl.transpose(R_wing_to_openvsp), wind_vector_in_body.vector), axis=wing_axis)
# print('Unit wind vector in wing axis: ', wind_vector_in_wing.vector.value)
alpha = csdl.arctan(wind_vector_in_wing.vector[2]/wind_vector_in_wing.vector.value[0])
# print('Effective angle of attack (deg): ', np.rad2deg(alpha.value))


## Aerodynamic Forces - from Modification IV
def calculate_forces_moments(force, moment, axis1, axis2):
    force_in_axis1 = Vector(vector=force, axis=axis1)
    moment_in_axis1 = Vector(vector=moment, axis=axis1)   
    force_moment_in_axis1 = ForcesMoments(force=force_in_axis1, moment=moment_in_axis1)
    force_moment_in_axis2 = force_moment_in_axis1.rotate_to_axis(axis2)
    return force_moment_in_axis1, force_moment_in_axis2

CL = 2*np.pi*alpha
e = 0.87
AR = 15
CD = 0.001 + 1/(np.pi*e*AR) * CL**2
rho = 1.225
S = 6.22
V = 35
L = 0.5*rho*V**2*CL*S
D = 0.5*rho*V**2*CD*S

aero_force = csdl.Variable(shape=(3, ), value=0.)
aero_moment = csdl.Variable(shape=(3, ), value=0.)
aero_force = aero_force.set(csdl.slice[0], -D)
aero_force = aero_force.set(csdl.slice[2], -L)

aero1, aero2 = calculate_forces_moments(force=aero_force, moment=aero_moment, axis1=wing_axis, axis2=fd_axis)
# print('Aero Force in Wing Axis: ', aero1.F.vector.value)
# print('Aero Moment in Wing Axis: ', aero1.M.vector.value)
# print('Aero Force in Body Axis: ', aero2.F.vector.value)
# print('Aero Moment in Body Axis: ', aero2.M.vector.value)



# Rotor Forces

cruise_motor_thrust=400
cruise_motor_prop_force = csdl.Variable(shape=(3, ), value=0.)
cruise_motor_prop_moment = csdl.Variable(shape=(3, ), value=0.)
cruise_motor_prop_force = cruise_motor_prop_force.set(csdl.slice[1], cruise_motor_thrust)

HL_motor_thrust = 200
HL_motor_prop_force = csdl.Variable(shape=(3, ), value=0.)
HL_motor_prop_moment = csdl.Variable(shape=(3, ), value=0.)
HL_motor_prop_force = HL_motor_prop_force.set(csdl.slice[1], HL_motor_thrust)

HL_props1 = []
HL_props2 = []
for i, HL_motor_axes in enumerate(HL_motor_axes, start=1):
    HL_prop1, HL_prop2 = calculate_forces_moments(force=HL_motor_prop_force, moment=cruise_motor_prop_moment,
                                                   axis1=HL_motor_axes, axis2=fd_axis)
    HL_props1.append(HL_prop1)
    HL_props2.append(HL_prop2)
    # print(f'High Lift Motor {i} - Prop Force in Cruise Motor Axis: ', HL_prop1.F.vector.value)
    # print(f'High Lift Motor {i} - Prop Moment in Cruise Motor Axis: ', HL_prop1.M.vector.value)
    # print(f'High Lift Motor {i} - Prop Force in Body Axis: ', HL_prop2.F.vector.value)
    # print(f'High Lift Motor {i} - Prop Moment in Body Axis: ', HL_prop2.M.vector.value)


cruise_props1 = []
cruise_props2 = []
for i, cruise_motor_axis in enumerate(cruise_motor_axes, start=1):
    cruise_prop1, cruise_prop2 = calculate_forces_moments(force=cruise_motor_prop_force, moment=cruise_motor_prop_moment,
                                                           axis1=cruise_motor_axis, axis2=fd_axis)
    cruise_props1.append(cruise_prop1)
    cruise_props2.append(cruise_prop2)
    # print(f'Cruise Motor {i} - Prop Force in Cruise Motor Axis: ', cruise_prop1.F.vector.value)
    # print(f'Cruise Motor {i} - Prop Moment in Cruise Motor Axis: ', cruise_prop1.M.vector.value)
    # print(f'Cruise Motor {i} - Prop Force in Body Axis: ', cruise_prop2.F.vector.value)
    # print(f'Cruise Motor {i} - Prop Moment in Body Axis: ', cruise_prop2.M.vector.value)





thrust_axis = cruise_motor1_tip - cruise_motor1_base
# print('Thrust Axis: ', thrust_axis.value)









from flight_simulator.core.vehicle.components.wing import Wing as WingComp
from flight_simulator.core.vehicle.components.fuselage import Fuselage as FuseComp
from flight_simulator.core.vehicle.components.aircraft import Aircraft as AircraftComp
from flight_simulator.core.vehicle.components.rotor import Rotor as RotorComp
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

parameterization_solver = ParameterizationSolver()
ffd_geometric_variables = GeometricVariables()

# Most of the values below come from the OpenVSP model
# lets vary AR from 12 - 18
# lets also 


Aircraft = AircraftComp(geometry=geometry, compute_surface_area_flag=False, parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)


# fuselage_length = csdl.Variable(name="fuselage_length", shape=(1, ), value=csdl.norm(fuselage_rear_guess[0] - fuselage_nose_guess[0]).value)
# Fuselage = FuseComp(
#     length=csdl.Variable(name="length", shape=(1, ), value=fuselage_length.value), geometry=fuselage, skip_ffd=False, parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)


# Aircraft.add_subcomponent(Fuselage)


wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=AR)
wing_span = csdl.Variable(name="wingspan", shape=(1, ), value=20)
wing_sweep = csdl.Variable(name="wing_sweep", shape=(1, ), value=35)


Wing = WingComp(AR=wing_AR,
                span=wing_span,
                sweep=wing_sweep,
                geometry=wing,
                parametric_geometry=wing_parametric_geometry,
                tight_fit_ffd=False, 
                orientation='horizontal', 
                name='Wing', parameterization_solver=parameterization_solver, ffd_geometric_variables=ffd_geometric_variables
                )

Aircraft.add_subcomponent(Wing)
wing_fuse_connection = geometry.evaluate(wing_te_center_parametric) - geometry.evaluate(fuselage_wing_te_center_parametric)
parameterization_solver.add_variable(computed_value=wing_fuse_connection[1:], desired_value=wing_fuse_connection.value[1:])

flapArea=left_flap_span*left_flap_chord
flapAR = left_flap_span**2/flapArea


flap_AR = csdl.Variable(name="flap_AR", shape=(1, ), value=flapAR)
flap_S_ref = csdl.Variable(name="flap_S_ref", shape=(1, ), value=flapArea)
flap_spanL = csdl.Variable(name="left_flap_span", shape=(1, ), value=2.5)
flap_spanR = csdl.Variable(name="right_flap_span", shape=(1, ), value=2.5)
flap_actuation_angle = csdl.Variable(name="flap_actuation_angle", shape=(1, ), value=0)

FlapsLeft = WingComp(AR=flap_AR, span=flap_spanL,
                                    geometry=flapL,parametric_geometry=left_flap_parametric_geometry,
                                    tight_fit_ffd=False, orientation="horizontal", name='Left Flap',
                                    parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)

FlapsRight = WingComp(AR=flap_AR, span=flap_spanR,
                                    geometry=flapR,parametric_geometry=right_flap_parametric_geometry,
                                    tight_fit_ffd=False, orientation="horizontal", name='Right Flap',
                                    parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)


aileronArea = left_aileron_span*left_aileron_chord
aileronAR = left_aileron_span**2/aileronArea
aileron_AR = csdl.Variable(name="aileron_AR", shape=(1, ), value=aileronAR)
aileron_S_ref = csdl.Variable(name="aileron_S_ref", shape=(1, ), value=aileronArea)
aileron_spanL = csdl.Variable(name="left_aileron_span", shape=(1, ), value=1.14)
aileron_spanR = csdl.Variable(name="right_aileron_span", shape=(1, ), value=1.14)
aileron_actuation_angle = csdl.Variable(name="aileron_actuation_angle", shape=(1, ), value=0)

Left_Aileron = WingComp(AR=aileron_AR, span=aileron_spanL,
                                    geometry=aileronL,parametric_geometry=left_aileron_parametric_geometry,
                                    tight_fit_ffd=False, name='Left Aileron',orientation='horizontal', 
                                    actuate_angle=aileron_actuation_angle, actuate_axis_location=0.,
                                    parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)

Right_Aileron = WingComp(AR=aileron_AR, span=aileron_spanR,
                                    geometry=aileronR,parametric_geometry=right_aileron_parametric_geometry,
                                    tight_fit_ffd=False, name='Right Aileron',orientation='horizontal', 
                                    actuate_angle=aileron_actuation_angle, actuate_axis_location=0.,
                                    parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)


Wing.add_subcomponent(FlapsLeft)
left_flap_wing_connection = geometry.evaluate(left_flap_le_center_parametric) - geometry.evaluate(left_flap_le_center_on_wing_te_parametric)
print("left_flap_wing_connection: ", left_flap_wing_connection.value)
parameterization_solver.add_variable(computed_value=left_flap_wing_connection, desired_value=left_flap_wing_connection.value)

Wing.add_subcomponent(FlapsRight)
right_flap_wing_connection = geometry.evaluate(right_flap_le_center_parametric) - geometry.evaluate(right_flap_le_center_on_wing_te_parametric)
# parameterization_solver.add_variable(computed_value=right_flap_wing_connection, desired_value=left_flap_wing_connection.value)

Wing.add_subcomponent(Left_Aileron)
left_aileron_wing_connection = geometry.evaluate(left_aileron_le_center_parametric) - geometry.evaluate(left_aileron_le_center_on_wing_te_parametric)
# parameterization_solver.add_variable(computed_value=left_aileron_wing_connection, desired_value=left_aileron_wing_connection.value)

Wing.add_subcomponent(Right_Aileron)
right_aileron_wing_connection = geometry.evaluate(right_aileron_le_center_parametric) - geometry.evaluate(right_aileron_le_center_on_wing_te_parametric)
# parameterization_solver.add_variable(computed_value=right_aileron_wing_connection, desired_value=right_aileron_wing_connection.value)



# HorTailArea = ht_span*ht_chord
# htAR = ht_span**2/HorTailArea
# TrimTabArea = trim_tab_span*trimTab_chord
# trimTabAR = trim_tab_span**2/TrimTabArea
# HorTail_AR = csdl.Variable(name="HT_AR", shape=(1, ), value=htAR)
# TrimTab_AR = csdl.Variable(name="TrimTab_AR", shape=(1, ), value=trimTabAR)
# HT_span = csdl.Variable(name="HT_span", shape=(1, ), value=3.14986972)
# TrimTab_span = csdl.Variable(name="TrimTab_span", shape=(1, ), value=1.9)
# HT_actuation_angle = csdl.Variable(name="HT_actuation_angle", shape=(1, ), value=0)

# HorTail = WingComp(AR=HorTail_AR, span=HT_span, 
#                    geometry=htALL, parametric_geometry=ht_parametric_geometry,
#                    tight_fit_ffd=False, skip_ffd=False,
#                    name='Horizontal Tail', orientation='horizontal', 
#                    parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)
# Aircraft.add_subcomponent(HorTail)

# tail_moment_arm_computed = csdl.norm(geometry.evaluate(ht_qc_center_parametric) - geometry.evaluate(wing_qc_center_parametric))
# print('Tail Moment Arm: ', tail_moment_arm_computed.value)
# parameterization_solver.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm_computed.value)

# TrimTab = WingComp(AR=TrimTab_AR, span=TrimTab_span, 
#                    geometry=trimTab, parametric_geometry=trimTab_parametric_geometry,
#                    tight_fit_ffd=False, name='Trim Tab', orientation='horizontal',
#                    parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)

# HorTail.add_subcomponent(TrimTab)


# VertTailArea = vt_span*vt_chord
# vtAR = vt_span**2/VertTailArea
# VT_span = csdl.Variable(name="VT_span", shape=(1, ), value=vt_span)


# RudderArea = rudder_span*rudder_chord
# rudderAR = rudder_span**2/RudderArea
# Rudder_span = csdl.Variable(name="Rudder_span", shape=(1, ), value=rudder_span)
# VT_actuation_angle = csdl.Variable(name="VT_actuation_angle", shape=(1, ), value=0)


# VertTail = WingComp(AR=vtAR, span=VT_span, geometry=vertTail, tight_fit_ffd=False, name='Vertical Tail', orientation='vertical',parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)
# Rudder = WingComp(AR=rudderAR, span=Rudder_span, geometry=rudder, tight_fit_ffd=False, name='Rudder', orientation='vertical', actuate_angle=VT_actuation_angle, actuate_axis_location=0.,parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)
# Aircraft.add_subcomponent(VertTail)
# VertTail.add_subcomponent(Rudder)

# rotors = Component(name='Rotors', 
#                    parameterization_solver=parameterization_solver,
#                    ffd_geometric_variables=ffd_geometric_variables)



# for i in range(1, 13):
#     HL_pylon = Component(name=f'Pylon {i}', geometry=eval(f'pylon{i}'),
#                          parameterization_solver=parameterization_solver,
#                          ffd_geometric_variables=ffd_geometric_variables)
#     Aircraft.add_subcomponent(HL_pylon)


# for i in range(1, 13):
#     HL_rotor = RotorComp(radius=MotorDisks[i-1],geometry=eval(f'spinner{i}'), 
#                          compute_surface_area_flag=False, skip_ffd=True, 
#                          name=f'HL Rotor {i}',
#                          parameterization_solver=parameterization_solver,
#                          ffd_geometric_variables=ffd_geometric_variables)
    

#     Aircraft.add_subcomponent(HL_rotor)
#     rotors_wing_connection = geometry.evaluate(eval(f'M{i}_disk_on_wing')) - geometry.evaluate(wing.project(eval(f'M{i}_disk_pt'), plot=False))
#     print(f'rotors_wing_connection {i}: ', rotors_wing_connection.value)
#     # parameterization_solver.add_variable(computed_value=rotors_wing_connection, desired_value=rotors_wing_connection.value)








# for i in range(1, 3):
#     CruiseRotor = RotorComp(radius=MotorDisks[i-1],
#                             geometry=eval(f'cruise_spinner{i}'), 
#                             compute_surface_area_flag=False, skip_ffd=True, 
#                             name=f'Cruise Rotor {i}',
#                             parameterization_solver=parameterization_solver,
#                             ffd_geometric_variables=ffd_geometric_variables)
#     Aircraft.add_subcomponent(CruiseRotor)



# cruise_motor_left_wing_connection = geometry.evaluate(wing_le_left_parametric) - geometry.evaluate(wing_on_cruise_motor1_parametric)
# print("cruise_motor_left_wing_connection: ", cruise_motor_left_wing_connection.value)
# cruise_motor_right_wing_connection = geometry.evaluate(wing_le_right_parametric) - geometry.evaluate(wing_on_cruise_motor2_parametric)
# print("cruise_motor_right_wing_connection: ", cruise_motor_right_wing_connection.value)
# parameterization_solver.add_variable(computed_value=cruise_motor_left_wing_connection, desired_value=cruise_motor_left_wing_connection.value)
# parameterization_solver.add_variable(computed_value=cruise_motor_right_wing_connection, desired_value=cruise_motor_right_wing_connection.value)


# geometry.plot()
parameterization_solver.evaluate(ffd_geometric_variables)




# plot_parameters = {
#     'camera': {
#         'pos': (15, 15 * 1.25, -12),
#         'focal_point': (-10 / 2, 0, 0),
#         'distance': 0,
#         'viewup': (0, 0, -1)
#     },
#     'screenshot': f'_x57_{5}.png',
#     'title': f'x57 Plot\nAspect Ratio {5}\nArea {25} m^2\nSpan {15} m',
# }

geometry.plot()








# from flight_simulator.core.vehicle.aircraft_control_system import AircraftControlSystem
# ControlSystem = AircraftControlSystem(symmetrical=False, airframe=Aircraft)       







recorder.stop()