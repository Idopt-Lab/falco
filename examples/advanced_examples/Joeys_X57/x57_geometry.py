import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt
import lsdo_geo as lg
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER, ureg, Q_
from flight_simulator.core.vehicle.component import Component
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from typing import Union, List
from dataclasses import dataclass
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.euler_rotations import build_rotation_matrix
from flight_simulator.core.vehicle.aircraft_control_system import AircraftControlSystem
from flight_simulator.core.vehicle.models.propulsion.propulsion_model import HLPropCurve, CruisePropCurve, AircraftPropulsion
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel, AircraftAerodynamics
from flight_simulator.core.vehicle.components.wing import Wing as WingComp
from flight_simulator.core.vehicle.components.fuselage import Fuselage as FuseComp
from flight_simulator.core.vehicle.components.aircraft import Aircraft as AircraftComp
from flight_simulator.core.vehicle.components.rotor import Rotor as RotorComp
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

lfs.num_workers = 1

debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=debug)

recorder.start()


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
h_tail = geometry.declare_component(function_search_names=['HorzStab'], name='h_tail')
trimTab = geometry.declare_component(function_search_names=['TrimTab'], name='trimTab')
htALL = geometry.declare_component(function_search_names=['HorzStab', 'TrimTab'], name='CompleteHT')
vertTail = geometry.declare_component(function_search_names=['VertTail'], name='vertTail')
rudder = geometry.declare_component(function_search_names=['Rudder'], name='rudder')
vtALL = geometry.declare_component(function_search_names=['VertTail', 'Rudder'], name='CompleteVT')
fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
gear_pod = geometry.declare_component(function_search_names=['GearPod'], name='gear_pod')

pylon1 = geometry.declare_component(function_search_names=['Pylon_12, 1'], name='pylon1')
pylon2 = geometry.declare_component(function_search_names=['Pylon_11, 1'], name='pylon2')
pylon3 = geometry.declare_component(function_search_names=['Pylon_10, 1'], name='pylon3')
pylon4 = geometry.declare_component(function_search_names=['Pylon_09, 1'], name='pylon4')
pylon5 = geometry.declare_component(function_search_names=['Pylon_08, 1'], name='pylon5')
pylon6 = geometry.declare_component(function_search_names=['Pylon_07, 1'], name='pylon6')
pylon7 = geometry.declare_component(function_search_names=['Pylon_07, 0'], name='pylon7')
pylon8 = geometry.declare_component(function_search_names=['Pylon_08, 0'], name='pylon8')
pylon9 = geometry.declare_component(function_search_names=['Pylon_09, 0'], name='pylon9')
pylon10 = geometry.declare_component(function_search_names=['Pylon_10, 0'], name='pylon10')
pylon11 = geometry.declare_component(function_search_names=['Pylon_11, 0'], name='pylon11')
pylon12 = geometry.declare_component(function_search_names=['Pylon_12, 0'], name='pylon12')

nacelle7 = geometry.declare_component(function_search_names=['HLNacelle_7_Tail'], name='nacelle7')
nacelle8 = geometry.declare_component(function_search_names=['HLNacelle_8_Tail'], name='nacelle8')
nacelle9 = geometry.declare_component(function_search_names=['HLNacelle_9_Tail'], name='nacelle9')
nacelle10 = geometry.declare_component(function_search_names=['HLNacelle_10_Tail'], name='nacelle10')
nacelle11 = geometry.declare_component(function_search_names=['HLNacelle_11_Tail'], name='nacelle11')
nacelle12 = geometry.declare_component(function_search_names=['HLNacelle_12_Tail'], name='nacelle12')

spinner1 = geometry.declare_component(function_search_names=['HL_Spinner12, 1'], name='spinner1')
spinner2 = geometry.declare_component(function_search_names=['HL_Spinner11, 1'], name='spinner2')
spinner3 = geometry.declare_component(function_search_names=['HL_Spinner10, 1'], name='spinner3')
spinner4 = geometry.declare_component(function_search_names=['HL_Spinner9, 1'], name='spinner4')
spinner5 = geometry.declare_component(function_search_names=['HL_Spinner8, 1'], name='spinner5')
spinner6 = geometry.declare_component(function_search_names=['HL_Spinner7, 1'], name='spinner6')
spinner7 = geometry.declare_component(function_search_names=['HL_Spinner7, 0'], name='spinner7')
spinner8 = geometry.declare_component(function_search_names=['HL_Spinner8, 0'], name='spinner8')
spinner9 = geometry.declare_component(function_search_names=['HL_Spinner9, 0'], name='spinner9')
spinner10 = geometry.declare_component(function_search_names=['HL_Spinner10, 0'], name='spinner10')
spinner11 = geometry.declare_component(function_search_names=['HL_Spinner11, 0'], name='spinner11')
spinner12 = geometry.declare_component(function_search_names=['HL_Spinner12, 0'], name='spinner12')

prop = geometry.declare_component(function_search_names=['HL-Prop'], name='prop')
motor = geometry.declare_component(function_search_names=['HL_Motor'], name='motor')
motor_interface = geometry.declare_component(function_search_names=['HL_Motor_Controller_Interface'], name='motor_interface')

cruise_spinner1 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 0'], name='cruise_spinner1')
cruise_spinner2 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 1'], name='cruise_spinner2')

wingALL = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4','Rt_Aileron','Lt_Aileron','Flap, 0','Flap, 1',
                                                            'Pylon_12, 1','Pylon_11, 1','Pylon_10, 1','Pylon_09, 1','Pylon_08, 1','Pylon_07, 1',
                                                            'Pylon_12, 0','Pylon_11, 0','Pylon_10, 0','Pylon_09, 0','Pylon_08, 0','Pylon_07, 0',
                                                            'HL_Spinner12, 1','HL_Spinner11, 1','HL_Spinner10, 1','HL_Spinner9, 1','HL_Spinner8, 1','HL_Spinner7, 1',
                                                            'HL_Spinner12, 0','HL_Spinner11, 0','HL_Spinner10, 0','HL_Spinner9, 0','HL_Spinner8, 0','HL_Spinner7, 0',
                                                            'CruiseNacelle-Spinner, 0','CruiseNacelle-Spinner, 1', 'Flap_Cover_7','Flap_Cover_9','Flap_Cover_11'], name='CompleteWing')

                                                     


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
left_aileron_qc = geometry.evaluate(left_aileron_qc_center_parametric)
left_aileron_qc_tip_right_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.4, -7.5])*ft2m, plot=False)
left_aileron_qc_tip_left_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -15.15, -7.5])*ft2m, plot=False)

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
right_aileron_qc = geometry.evaluate(right_aileron_qc_center_parametric)
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
left_flap_qc = geometry.evaluate(left_flap_qc_center_parametric)
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
right_flap_qc = geometry.evaluate(right_flap_qc_center_parametric)
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
ht_qc = geometry.evaluate(ht_qc_center_parametric)
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
vt_qc_tip_parametric = vertTail.project(np.array([-27.33 + (0.25*(-28.7+23)), 0., -10.85])*ft2m,plot=False)


vt_span = np.linalg.norm(vt_te_base.value - vt_te_tip.value)
# print('Vertical Tail Span: ', vt_span)

vt_chord = np.linalg.norm(vt_le_mid.value - vt_te_mid.value)
# print('Vertical Tail Chord: ', vt_chord)

rudder_le_base_parametric = rudder.project(np.array([-23, 0, -5.5])*ft2m, plot=False)
rudder_le_base = geometry.evaluate(rudder_le_base_parametric)
rudder_le_mid_parametric = rudder.project(np.array([-28.7, 0., -8.5])*ft2m, plot=False)
rudder_le_mid = geometry.evaluate(rudder_le_mid_parametric)
rudder_le_tip = geometry.evaluate(rudder.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False))

rudder_te_base_parametric = rudder.project(np.array([-29.5, 0, -5.5])*ft2m, plot=False)
rudder_te_mid_parametric = rudder.project(np.array([-30., 0., -8.5])*ft2m, plot=False)
rudder_te_tip_parametric = rudder.project(np.array([-30.4, 0, -11.])*ft2m, plot=False)


rudder_span = np.linalg.norm(rudder_le_base.value - rudder_le_tip.value)
# print('Rudder Span: ', rudder_span)
rudder_chord = np.linalg.norm(rudder_le_mid.value - rudder_le_tip.value)
# print('Rudder Chord: ', rudder_chord)

vt_parametric_geometry = [
    vt_le_base_parametric,
    vt_le_tip_parametric,
    vt_le_base_parametric,
    rudder_te_base_parametric,
    rudder_te_tip_parametric,
    rudder_te_mid_parametric,
    vt_qc_parametric,
    vt_qc_base_parametric,
    vt_qc_tip_parametric
]

# Fuselage Region Info
fuselage_wing_le_center_parametric = fuselage.project(np.array([-12.356, 0., -5.5])*ft2m, plot=False)
fuselage_wing_qc_center_parametric = fuselage.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False)
fuselage_wing_qc = geometry.evaluate(fuselage_wing_qc_center_parametric)
fuselage_wing_te_center_parametric = fuselage.project(np.array([-14.25, 0., -5.5])*ft2m, plot=False)
fuselage_wing_te_center = geometry.evaluate(fuselage_wing_te_center_parametric)
fuselage_tail_qc = geometry.evaluate(fuselage.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))
fuselage_tail_te_center_parametric = fuselage.project(np.array([-30, 0., -5.5])*ft2m, plot=False)
fuselage_tail_te_center = geometry.evaluate(fuselage_tail_te_center_parametric)


# Propeller Region Info
M12_disk_pt =  np.array([-12.5, 14, -7.355])*ft2m
M1_disk_pt = np.array([-12.5, -14, -7.355])*ft2m
M11_disk_pt =  np.array([-12.35, 12, -7.355])*ft2m
M2_disk_pt = np.array([-12.35, -12, -7.355])*ft2m
M10_disk_pt = np.array([-12.2, 10, -7.659])*ft2m
M3_disk_pt = np.array([-12.2, -10, -7.659])*ft2m
M9_disk_pt = np.array([-12, 8, -7.659])*ft2m
M4_disk_pt = np.array([-12, -8, -7.659])*ft2m
M8_disk_pt = np.array([-11.8, 6, -7.659])*ft2m
M5_disk_pt = np.array([-11.8, -6, -7.659])*ft2m
M7_disk_pt = np.array([-11.6, 4, -7.659])*ft2m
M6_disk_pt = np.array([-11.6, -4, -7.659])*ft2m

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


# Define pylon connection points
M1_pylon_pt = np.array([-12.5, -14, -7.355])*ft2m 
M2_pylon_pt = np.array([-12.35, -12, -7.355])*ft2m
M3_pylon_pt = np.array([-12.2, -10, -7.659])*ft2m
M4_pylon_pt = np.array([-12, -8, -7.659])*ft2m
M5_pylon_pt = np.array([-11.8, -6, -7.659])*ft2m
M6_pylon_pt = np.array([-11.6, -4, -7.659])*ft2m
M7_pylon_pt = np.array([-11.6, 4, -7.659])*ft2m
M8_pylon_pt = np.array([-11.8, 6, -7.659])*ft2m
M9_pylon_pt = np.array([-12, 8, -7.659])*ft2m
M10_pylon_pt = np.array([-12.2, 10, -7.659])*ft2m
M11_pylon_pt = np.array([-12.35, 12, -7.355])*ft2m
M12_pylon_pt = np.array([-12.5, 14, -7.355])*ft2m

# Project pylon points onto their respective components
M1_pylon_on_wing = pylon1.project(M1_pylon_pt, plot=False)
M2_pylon_on_wing = pylon2.project(M2_pylon_pt, plot=False)
M3_pylon_on_wing = pylon3.project(M3_pylon_pt, plot=False)
M4_pylon_on_wing = pylon4.project(M4_pylon_pt, plot=False)
M5_pylon_on_wing = pylon5.project(M5_pylon_pt, plot=False)
M6_pylon_on_wing = pylon6.project(M6_pylon_pt, plot=False)
M7_pylon_on_wing = pylon7.project(M7_pylon_pt, plot=False)
M8_pylon_on_wing = pylon8.project(M8_pylon_pt, plot=False)
M9_pylon_on_wing = pylon9.project(M9_pylon_pt, plot=False)
M10_pylon_on_wing = pylon10.project(M10_pylon_pt, plot=False)
M11_pylon_on_wing = pylon11.project(M11_pylon_pt, plot=False)
M12_pylon_on_wing = pylon12.project(M12_pylon_pt, plot=False)




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

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    # OpenVSP Model Axis
    openvsp_axis = Axis(
        name='OpenVSP Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        origin=ValidOrigins.OpenVSP.value

    )
    
    wing_axis = AxisLsdoGeo(
        name='Wing Axis',
        geometry=wing,
        parametric_coords=wing_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
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
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_flap_axis = AxisLsdoGeo(
        name='Right Flap Axis',
        geometry=flapR,
        parametric_coords=right_flap_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    left_aileron_axis = AxisLsdoGeo(
        name='Left Aileron Axis',
        geometry=aileronL,
        parametric_coords=left_aileron_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_aileron_axis = AxisLsdoGeo(
        name='Right Aileron Axis',
        geometry=aileronR,
        parametric_coords=right_aileron_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Tail Region Axis

    ht_tail_axis = AxisLsdoGeo(
        name='Horizontal Tail Axis',
        geometry=h_tail,
        parametric_coords=ht_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
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
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=ht_tail_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    # print('Trim Tab axis translation (ft): ', trimTab_axis.translation.value)
    # print('Trim Tab axis rotation (deg): ', np.rad2deg(trimTab_axis.euler_angles_vector.value))
    # geometry.plot()

    vt_tail_axis = AxisLsdoGeo(
        name='Vertical Tail Axis',
        geometry=rudder,
        parametric_coords=rudder_le_mid_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
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
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
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
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
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
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M4_axis = AxisLsdoGeo(
        name= 'Motor 4 Axis',
        geometry=spinner4,
        parametric_coords=M4_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M5_axis = AxisLsdoGeo(
        name= 'Motor 5 Axis',
        geometry=spinner5,
        parametric_coords=M5_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M6_axis = AxisLsdoGeo(
        name= 'Motor 6 Axis',
        geometry=spinner6,
        parametric_coords=M6_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M7_axis = AxisLsdoGeo(
        name= 'Motor 7 Axis',
        geometry=spinner7,
        parametric_coords=M7_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M8_axis = AxisLsdoGeo(
        name= 'Motor 8 Axis',
        geometry=spinner8,
        parametric_coords=M8_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M9_axis = AxisLsdoGeo(
        name= 'Motor 9 Axis',
        geometry=spinner9,
        parametric_coords=M9_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M10_axis = AxisLsdoGeo(
        name= 'Motor 10 Axis',
        geometry=spinner10,
        parametric_coords=M10_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M11_axis = AxisLsdoGeo(
        name= 'Motor 11 Axis',
        geometry=spinner11,
        parametric_coords=M11_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M12_axis = AxisLsdoGeo(
        name= 'Motor 12 Axis',
        geometry=spinner12,
        parametric_coords=M12_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Right Outermost motor axis translation (ft): ', r_om_axis.translation.value)
    # print('Right Outermost motor axis rotation (deg): ', np.rad2deg(r_om_axis.euler_angles_vector.value))

    HL_motor_axes = [M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis]
    # Cruise Motor Region


    cruise_motor1_axis = AxisLsdoGeo(
        name= 'Cruise Motor 1 Axis',
        geometry=cruise_spinner1,
        parametric_coords=cruise_motor1_tip_parametric,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
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
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Cruise motor axis translation (ft): ', cruise_motor_axis.translation.value)
    # print('Cruise motor axis rotation (deg): ', np.rad2deg(cruise_motor_axis.euler_angles_vector.value))

    cruise_motor_axes = [cruise_motor1_axis, cruise_motor2_axis]


    fd_axis = Axis(
        name='Flight Dynamics Body Fixed Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )
    # print('Body-fixed angles (deg)', np.rad2deg(fd_axis.euler_angles_vector.value))



    wind_axis = Axis(
        name='Wind Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )
    # print('Wind axis angles (deg)', np.rad2deg(wind_axis.euler_angles_vector.value))


    return openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geometry, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis

openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis,geometry,left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis = axes_create()




### FORCES AND MOMENTS MODELLING


x_57_states = AircraftStates(axis=fd_axis,u=Q_(172, 'mph')) # stall speed
x_57_mi = MassMI(axis=fd_axis,
                 Ixx=Q_(4314.08, 'kg*(m*m)'),
                 Ixy=Q_(-232.85, 'kg*(m*m)'),
                 Ixz=Q_(-2563.29, 'kg*(m*m)'),
                 Iyy=Q_(18656.93, 'kg*(m*m)'),
                 Iyz=Q_(-62.42, 'kg*(m*m)'),
                 Izz=Q_(22340.21, 'kg*(m*m)'),
                 )

x57_mass_properties = MassProperties(mass=Q_(1360.77, 'kg'),
                                      inertia=x_57_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))


x57_controls = AircraftControlSystem(engine_count=12,symmetrical=True)
x57_aircraft = Component(name='X-57')
x57_aircraft.quantities.mass_properties = x57_mass_properties
HL_radius_x57 = csdl.Variable(shape=(1,), value=2*ft2m) # HL propeller radius in meters
cruise_radius_x57 = csdl.Variable(shape=(1,), value=5*ft2m) # cruise propeller radius in meters

e_x57 = csdl.Variable(shape=(1,), value=0.87) # Oswald efficiency factor
CD0_x57 = csdl.Variable(shape=(1,), value=0.001) # Zero-lift drag coefficient
S_x57 = csdl.Variable(shape=(1,), value=66.667*(ft2m**2)) # Wing area in m^2
b_x57 = csdl.Variable(shape=(1,), value=31.623*ft2m) # Wing span in meters
AR_x57 = b_x57**2/S_x57 # Aspect ratio of the wing
incidence_x57 = csdl.Variable(shape=(1,), value=2*np.pi/180) # Wing incidence angle in radians



## Aerodynamic Forces - from Modification IV
x_57_wing_state = AircraftStates(
    axis=wing_axis,u=Q_(172, 'mph'))

atmospheric_states = x_57_states.atmospheric_states
x57_lift_model = LiftModel(AR=AR_x57, e=e_x57, CD0=CD0_x57, S=S_x57, incidence=incidence_x57)
x57_aero = AircraftAerodynamics(states=x_57_wing_state, controls=x57_controls, lift_model=x57_lift_model,atmospheric_states=atmospheric_states)
aero_loads1 = x57_aero.get_FM_refPoint()
aero_loads2 = aero_loads1.rotate_to_axis(fd_axis)

print('\nAerodynamic Forces and Moments:')
print('-' * 40)
print(f'Forces in Wing Axis: [{aero_loads1.F.vector.value[0]:.2f}, {aero_loads1.F.vector.value[1]:.2f}, {aero_loads1.F.vector.value[2]:.2f}] N')
print(f'Moments in Wing Axis: [{aero_loads1.M.vector.value[0]:.2f}, {aero_loads1.M.vector.value[1]:.2f}, {aero_loads1.M.vector.value[2]:.2f}] N⋅m')
print(f'Forces in Flight Dynamics Axis: [{aero_loads2.F.vector.value[0]:.2f}, {aero_loads2.F.vector.value[1]:.2f}, {aero_loads2.F.vector.value[2]:.2f}] N')
print(f'Moments in Flight Dynamics Axis: [{aero_loads2.M.vector.value[0]:.2f}, {aero_loads2.M.vector.value[1]:.2f}, {aero_loads2.M.vector.value[2]:.2f}] N⋅m')
print('-' * 40)


x57_aero.plot_aerodynamics(
    velocity_range=(20, 200),
    alpha_range=(-5, 15),
    AR_values=[15],
    e_values=[0.75, 0.85, 0.95]
)





# Rotor Forces


HL_motor_states = []
HL_motor_props = []
HL_prop_loads_list = []
HL_prop_loads2_list = []

for i, motor_axis in enumerate(HL_motor_axes):
    motor_state = AircraftStates(
        axis=motor_axis,
        u=Q_(172, 'mph')
    )
    HL_motor_states.append(motor_state)
    
    prop_curve = HLPropCurve()
    
    motor_prop = AircraftPropulsion(
        states=motor_state,
        controls=x57_controls, 
        radius=HL_radius_x57, 
        prop_curve=prop_curve
    )
    HL_motor_props.append(motor_prop)
    
    prop_load = motor_prop.get_FM_refPoint()
    HL_prop_loads_list.append(prop_load)
    prop_load2 = prop_load.rotate_to_axis(fd_axis)
    HL_prop_loads2_list.append(prop_load2)
    
    print(f'Motor {i+1} Forces and Moments:')
    print('-' * 40)
    print(f'Forces in Motor {i+1} Axis: [{prop_load.F.vector.value[0]:.2f}, {prop_load.F.vector.value[1]:.2f}, {prop_load.F.vector.value[2]:.2f}] N')
    print(f'Moments in Motor {i+1} Axis: [{prop_load.M.vector.value[0]:.2f}, {prop_load.M.vector.value[1]:.2f}, {prop_load.M.vector.value[2]:.2f}] N⋅m')
    print(f'Forces in Flight Dynamics Axis: [{prop_load2.F.vector.value[0]:.2f}, {prop_load2.F.vector.value[1]:.2f}, {prop_load2.F.vector.value[2]:.2f}] N')
    print(f'Moments in Flight Dynamics Axis: [{prop_load2.M.vector.value[0]:.2f}, {prop_load2.M.vector.value[1]:.2f}, {prop_load2.M.vector.value[2]:.2f}] N⋅m')
    print('-' * 40)



# Cruise Motor Forces
cruise_motor_states = []
cruise_motor_props = []
cruise_prop_loads_list = []
cruise_prop_loads2_list = []

for i, cruise_motor_axis in enumerate(cruise_motor_axes):
    cruise_motor_state = AircraftStates(
        axis=cruise_motor_axis,
        u=Q_(172, 'mph')
    )
    cruise_motor_states.append(cruise_motor_state)
    
    cruise_prop_curve = CruisePropCurve()
    
    cruise_motor_prop = AircraftPropulsion(
        states=cruise_motor_state,
        controls=x57_controls, 
        radius=cruise_radius_x57,
        prop_curve=cruise_prop_curve
    )
    cruise_motor_props.append(cruise_motor_prop)
    
    cruise_prop_load = cruise_motor_prop.get_FM_refPoint()
    cruise_prop_loads_list.append(cruise_prop_load)
    cruise_prop_load2 = cruise_prop_load.rotate_to_axis(fd_axis)
    cruise_prop_loads2_list.append(cruise_prop_load2)
    
    print(f'Cruise Motor {i+1} Forces and Moments:')
    print('-' * 40)
    print(f'Forces in Cruise Motor {i+1} Axis: [{cruise_prop_load.F.vector.value[0]:.2f}, {cruise_prop_load.F.vector.value[1]:.2f}, {cruise_prop_load.F.vector.value[2]:.2f}] N')
    print(f'Moments in Cruise Motor {i+1} Axis: [{cruise_prop_load.M.vector.value[0]:.2f}, {cruise_prop_load.M.vector.value[1]:.2f}, {cruise_prop_load.M.vector.value[2]:.2f}] N⋅m')
    print(f'Forces in Flight Dynamics Axis: [{cruise_prop_load2.F.vector.value[0]:.2f}, {cruise_prop_load2.F.vector.value[1]:.2f}, {cruise_prop_load2.F.vector.value[2]:.2f}] N')
    print(f'Moments in Flight Dynamics Axis: [{cruise_prop_load2.M.vector.value[0]:.2f}, {cruise_prop_load2.M.vector.value[1]:.2f}, {cruise_prop_load2.M.vector.value[2]:.2f}] N⋅m')
    print('-' * 40)



HL_motor_props[1].plot_propulsion(
    velocity_range=(0, 200),
    # ref_velocities=[50, 75, 80],  # Compare different reference velocities
    # ref_throttles=[0.3, 0.5, 0.7],  # Compare different throttle settings
    rpm_ranges=[(1000,4702)],  # Compare different RPM ranges
    radius_values=[1*ft2m, 2*ft2m, 3*ft2m],  # Different radii in meters (1ft, 2ft, 3ft)
    ref_throttles=[1.0, 1.0, 1.0],  # Different throttle settings
    labels=['1ft', '2ft', '3ft'],  # Custom labels for each configuration
    title=f'High-Lift Motor {1}'
)


cruise_motor_props[1].plot_propulsion(
    velocity_range=(0, 200),
    # ref_velocities=[50, 75, 80],  # Compare different reference velocities
    # ref_throttles=[0.3, 0.5, 0.7],  # Compare different throttle settings
    rpm_ranges=[(1000,2250)],  # Compare different RPM ranges
    radius_values=[4*ft2m, 5*ft2m, 6*ft2m],  # Different radii in meters (4ft, 5ft, 6ft)
    ref_throttles=[1.0, 1.0, 1.0],  # Different throttle settings
    labels=['4ft', '5ft', '6ft'],  # Custom labels for each configuration
    title=f'Cruise Motor {1}'
)
plt.show()


thrust_axis = cruise_motor1_tip - cruise_motor1_base
# print('Thrust Axis: ', thrust_axis.value)




total_prop_forces = np.zeros(3)
total_prop_moments = np.zeros(3)


for i, prop_load in enumerate(HL_prop_loads2_list):
    total_prop_forces += prop_load.F.vector.value
    total_prop_moments += prop_load.M.vector.value


for i, cruise_load in enumerate(cruise_prop_loads2_list):
    total_prop_forces += cruise_load.F.vector.value
    total_prop_moments += cruise_load.M.vector.value

aero_forces = aero_loads2.F.vector.value
aero_moments = aero_loads2.M.vector.value

total_forces = aero_forces + total_prop_forces
total_moments = aero_moments + total_prop_moments

print('\nTotal Aircraft Loads in Flight Dynamics Axis:')
print('-' * 40)
print('Forces:')
print(f'Aerodynamic Forces: [{aero_forces[0]:.2f}, {aero_forces[1]:.2f}, {aero_forces[2]:.2f}] N')
print(f'Propulsive Forces: [{total_prop_forces[0]:.2f}, {total_prop_forces[1]:.2f}, {total_prop_forces[2]:.2f}] N')
print(f'Net Total Forces: [{total_forces[0]:.2f}, {total_forces[1]:.2f}, {total_forces[2]:.2f}] N')
print('\nMoments:')
print(f'Aerodynamic Moments: [{aero_moments[0]:.2f}, {aero_moments[1]:.2f}, {aero_moments[2]:.2f}] N⋅m')
print(f'Propulsive Moments: [{total_prop_moments[0]:.2f}, {total_prop_moments[1]:.2f}, {total_prop_moments[2]:.2f}] N⋅m')
print(f'Net Total Moments: [{total_moments[0]:.2f}, {total_moments[1]:.2f}, {total_moments[2]:.2f}] N⋅m')
print('-' * 40)







## Aircraft Component Creation

parameterization_solver = ParameterizationSolver()
ffd_geometric_variables = GeometricVariables()




Aircraft = AircraftComp(geometry=geometry, compute_surface_area_flag=False, 
                        parameterization_solver=parameterization_solver,
                        ffd_geometric_variables=ffd_geometric_variables)



Fuselage = FuseComp(
    length=csdl.Variable(name="length", shape=(1, ), value=8.2242552),
    max_height=csdl.Variable(name="max_height", shape=(1, ), value=1.09236312),
    max_width=csdl.Variable(name="max_width", shape=(1, ), value=1.24070602),
    geometry=fuselage, skip_ffd=False, 
    parameterization_solver=parameterization_solver,
    ffd_geometric_variables=ffd_geometric_variables)

# print(f"Fuselage parameters: {Fuselage.parameters.__dict__}")

Aircraft.add_subcomponent(Fuselage)



aileronL.rotate(left_aileron_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
aileronR.rotate(right_aileron_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
flapL.rotate(left_flap_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
flapR.rotate(right_flap_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
htALL.rotate(ht_qc, np.array([0., 1., 0.]), angles=np.deg2rad(0))
rudder.rotate(rudder_le_mid, np.array([0., 0., 1.]), angles=np.deg2rad(0))



wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=15)
wing_span = csdl.Variable(name="wingspan", shape=(1, ), value=9.6)
wing_sweep = csdl.Variable(name="wing_sweep", shape=(1, ), value=0)
wing_dihedral = csdl.Variable(name="wing_dihedral", shape=(1, ), value=0)

Wing = WingComp(AR=wing_AR,
                span=wing_span,
                sweep=wing_sweep,
                dihedral=wing_dihedral,
                geometry=wingALL,
                parametric_geometry=wing_parametric_geometry,
                tight_fit_ffd=False, 
                orientation='horizontal', 
                name='Wing', parameterization_solver=parameterization_solver, 
                ffd_geometric_variables=ffd_geometric_variables
                )

Aircraft.add_subcomponent(Wing)
# wing_le_fuse_connection = geometry.evaluate(wing_le_center_parametric) - geometry.evaluate(fuselage_wing_le_center_parametric)
# wing_te_fuse_connection = geometry.evaluate(wing_te_center_parametric) - geometry.evaluate(fuselage_wing_te_center_parametric)
wing_qc_fuse_connection = geometry.evaluate(wing_qc_center_parametric) - geometry.evaluate(fuselage_wing_qc_center_parametric)
# print("wing_fuse_connection: ", wing_fuse_connection.value)
# parameterization_solver.add_variable(computed_value=wing_le_fuse_connection, desired_value=wing_le_fuse_connection.value)
parameterization_solver.add_variable(computed_value=wing_qc_fuse_connection, desired_value=wing_qc_fuse_connection.value)
# parameterization_solver.add_variable(computed_value=wing_te_fuse_connection, desired_value=wing_te_fuse_connection.value)




HorTailArea = ht_span*ht_chord
htAR = ht_span**2/HorTailArea
HorTail_AR = csdl.Variable(name="HT_AR", shape=(1, ), value=4)
HT_span = csdl.Variable(name="HT_span", shape=(1, ), value=3.14986972)
HT_sweep = csdl.Variable(name="HT_sweep", shape=(1, ), value=0)

HorTail = WingComp(AR=HorTail_AR, span=HT_span, sweep=HT_sweep,
                   geometry=htALL, parametric_geometry=ht_parametric_geometry,
                   tight_fit_ffd=False, skip_ffd=False,
                   name='Horizontal Tail', orientation='horizontal', 
                   parameterization_solver=parameterization_solver,ffd_geometric_variables=ffd_geometric_variables)
Aircraft.add_subcomponent(HorTail)


tail_moment_arm_computed = csdl.norm(geometry.evaluate(ht_qc_center_parametric) - geometry.evaluate(wing_qc_center_parametric))
h_tail_fuselage_connection = geometry.evaluate(ht_te_center_parametric) - geometry.evaluate(fuselage_tail_te_center_parametric)
# print('Tail Moment Arm: ', tail_moment_arm_computed.value)
# print('Tail Fuselage Connection: ', h_tail_fuselage_connection.value)
# parameterization_solver.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm_computed.value)
parameterization_solver.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)



vt_AR= csdl.Variable(name="VT_AR", shape=(1, ), value=1.998)
VT_span = csdl.Variable(name="VT_span", shape=(1, ), value=2.3761728)
VT_actuation_angle = csdl.Variable(name="VT_actuation_angle", shape=(1, ), value=0)
VT_sweep = csdl.Variable(name="VT_sweep", shape=(1, ), value=-40)


# VertTail = WingComp(AR=vt_AR, span=VT_span, sweep=VT_sweep,
#                     geometry=vtALL, parametric_geometry=vt_parametric_geometry,
#                     tight_fit_ffd=False, 
#                     name='Vertical Tail', orientation='vertical',
#                     parameterization_solver=parameterization_solver,
#                     ffd_geometric_variables=ffd_geometric_variables)
# Aircraft.add_subcomponent(VertTail)

vtail_fuselage_connection = geometry.evaluate(fuselage_rear_pts_parametric) - geometry.evaluate(vt_qc_base_parametric)
# print('VTail Fuselage Connection: ', vtail_fuselage_connection.value)
parameterization_solver.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)



# geometry.plot()
parameterization_solver.evaluate(ffd_geometric_variables)
geometry.plot(camera=dict(pos=(12, 15, -12),  # Camera position 
                         focal_point=(-Fuselage.parameters.length.value/2, 0, 0),  # Point camera looks at
                         viewup=(0, 0, -1)),    # Camera up direction
                        #  title= f'X-57 Maxwell Aircraft Geometry\nWing Span: {Wing.parameters.span.value[0]:.2f} m\nWing AR: {Wing.parameters.AR.value[0]:.2f}\nWing Area S: {Wing.parameters.S_ref.value[0]:.2f} m^2\nWing Sweep: {Wing.parameters.sweep.value[0]:.2f} deg\nAileron Deflection: {aileron_actuation_angle.value[0]:.2f} deg\nFlap Deflection: {flap_actuation_angle.value[0]:.2f} deg\nHorizontal Tail Deflection: {HT_actuation_angle.value[0]:.2f} deg\nRudder Deflection: {rudder_actuation_angle.value[0]:.2f} deg',
                         title=f'X-57 Maxwell Aircraft Geometry\nFuselage Length: {Fuselage.parameters.length.value[0]:.2f} m\nFuselage Height: {Fuselage.parameters.max_height.value[0]:.2f} m\nFuselage Width: {Fuselage.parameters.max_width.value[0]:.2f} m',
                         screenshot= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57'/ 'images' / f'x_57_{Wing.parameters.span.value[0]}_AR_{Wing.parameters.AR.value[0]}_S_ref_{Wing.parameters.S_ref.value[0]}_sweep_{Wing.parameters.sweep.value[0]}.png')





recorder.stop()